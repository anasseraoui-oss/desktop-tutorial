# Anaconda officielle (Jupyter Notebook déjà inclus)
FROM continuumio/anaconda3:latest
WORKDIR /opt/notebooks
COPY notebooks/ /opt/notebooks/
EXPOSE 8888
CMD ["jupyter", "notebook", "--notebook-dir=/opt/notebooks", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
