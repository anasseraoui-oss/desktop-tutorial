"""Carte auto-organisatrice de Kohonen (SOM) en NumPy.

Même interface que le module utilisé dans le TP :
    som = SOM(); som.create(width, height, ch)
    som.fit(dataset, random_sampling, n_iter, dist_method)
    som.find_maching_nodes(dataset) -> (locations, distances)
    som.get_map_vectors(), som.get_umatrix(), som.get_component_plane(i)
    plot_data_on_map(umatrix, data_locations, data_colors, node_width, data_marker_size)
"""
import sys
import numpy as np
import matplotlib.pyplot as plt


def _progress(label, i, n, width=20):
    done = int(width * (i + 1) / n)
    bar = '=' * done + ('>' if done < width else '')
    sys.stdout.write('\r{} [{:<{w}}] {}'.format(label, bar, 'Done' if i + 1 == n else '', w=width + 1))
    if i + 1 == n:
        sys.stdout.write('\n')
    sys.stdout.flush()


class SOM:
    def __init__(self):
        self.map = None

    def create(self, width, height, ch):
        """Crée une carte width x height de vecteurs de dimension ch, initialisés aléatoirement."""
        self.width, self.height, self.ch = width, height, ch
        self.map = np.random.rand(height, width, ch).astype(np.float32)
        ys, xs = np.mgrid[0:height, 0:width]
        self._grid = np.stack([ys, xs], axis=-1).astype(np.float32)

    def _distances(self, vector, dist_method):
        if dist_method == 'euclidean':
            return np.sqrt(np.sum((self.map - vector) ** 2, axis=-1))
        if dist_method == 'cosine':
            num = np.sum(self.map * vector, axis=-1)
            den = np.linalg.norm(self.map, axis=-1) * np.linalg.norm(vector) + 1e-12
            return 1 - num / den
        raise ValueError("dist_method doit être 'euclidean' ou 'cosine'")

    def _bmu(self, vector, dist_method):
        d = self._distances(vector, dist_method)
        idx = np.unravel_index(np.argmin(d), d.shape)
        return idx, d[idx]

    def fit(self, dataset, random_sampling=1.0, n_iter=20, dist_method='euclidean',
            learning_rate=0.5, sigma=None):
        """Entraîne la carte.

        random_sampling : fraction (0..1] du dataset tirée au hasard à chaque itération.
        Le taux d'apprentissage et le rayon du voisinage gaussien décroissent au fil des itérations.
        """
        assert 0 < random_sampling <= 1, 'random_sampling doit être dans ]0, 1]'
        dataset = np.asarray(dataset, dtype=np.float32)
        # Initialise la carte dans l'étendue des données
        lo, hi = dataset.min(axis=0), dataset.max(axis=0)
        self.map = lo + self.map * (hi - lo)
        sigma0 = sigma or max(self.width, self.height) / 2
        n_samples = max(1, int(len(dataset) * random_sampling))
        for it in range(n_iter):
            frac = it / n_iter
            lr = learning_rate * (1 - frac)
            sig = sigma0 * (1 - frac) + 0.5
            for v in dataset[np.random.choice(len(dataset), n_samples, replace=False)]:
                (by, bx), _ = self._bmu(v, dist_method)
                g2 = np.sum((self._grid - (by, bx)) ** 2, axis=-1)
                h = np.exp(-g2 / (2 * sig ** 2))[..., None]
                self.map += lr * h * (v - self.map)
            _progress('Training SOM', it, n_iter)
        self.dist_method = dist_method

    def find_maching_nodes(self, dataset):
        """Pour chaque vecteur : position (ligne, colonne) du meilleur nœud et distance."""
        dist_method = getattr(self, 'dist_method', 'euclidean')
        locations, distances = [], []
        n = len(dataset)
        for i, v in enumerate(dataset):
            idx, d = self._bmu(v, dist_method)
            locations.append(idx)
            distances.append(d)
            if (i + 1) % max(1, n // 20) == 0 or i + 1 == n:
                _progress('Finding mathing nodes', i, n)
        return np.array(locations), np.array(distances)

    def get_map_vectors(self):
        return self.map

    def get_umatrix(self):
        """Distance moyenne de chaque nœud à ses voisins (4-connexité)."""
        h, w, _ = self.map.shape
        um = np.zeros((h, w))
        for y in range(h):
            for x in range(w):
                ds = [np.linalg.norm(self.map[y, x] - self.map[ny, nx])
                      for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))
                      if 0 <= ny < h and 0 <= nx < w]
                um[y, x] = np.mean(ds)
        return um

    def get_component_plane(self, component):
        return self.map[:, :, component]


def plot_data_on_map(umatrix, data_locations, data_colors=None, node_width=20,
                     data_marker_size=20, invert_umatrix=True, filename='som_data_on_map.png'):
    """Affiche la U-matrice et superpose les vecteurs de données ; sauvegarde en PNG."""
    h, w = umatrix.shape
    img = umatrix.max() - umatrix if invert_umatrix else umatrix
    fig = plt.figure(figsize=(w * node_width / 100, h * node_width / 100))
    plt.imshow(img, cmap='gray', interpolation='nearest')
    jitter = (np.random.rand(len(data_locations), 2) - 0.5) * 0.6
    plt.scatter(data_locations[:, 1] + jitter[:, 0], data_locations[:, 0] + jitter[:, 1],
                c=data_colors, s=data_marker_size, edgecolors='k', linewidths=0.3)
    plt.axis('off')
    plt.title('Données projetées sur la U-matrice')
    plt.savefig(filename, bbox_inches='tight', dpi=100)
    plt.show()
    return fig
