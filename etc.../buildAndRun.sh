docker build -t recommender . 
&& docker rm recommender 
&& docker run --name recommender --gpus all -it -v .:/workdir recommender