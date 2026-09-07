~~~bash
docker save -o myapp.tar myapp:latest
docker load -i myapp.tar

docker save myapp:latest | gzip > myapp.tar.gz
docker load < myapp.tar.gz
~~~