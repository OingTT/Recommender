# Build Docker Image
docker build -t \<Image-Name> \<Docker-File-Path>

docker build -t recommender .

# Remove Docker already running Container which had same name
docker rm \<Container-Name>

docker rm recommender

# Run Docker Container
* -it Flag => Interactive
* -p [Port1]:[Port2] Flag => Bind Port1 to Port2
* -v \<Host-Directory>:\<Container-Directory>

docker run --name \<Container-Name> -it \<Image-Name>

docker run --name recommender -it recommender

# Use Gpu Acceleration
docker run --name \<Container-Name> --gpus all -it \<Image-Name>

docker run --name recommender --gpus all -it recommender

# Remove and Run Docker Container
docker rm \<Container-Name> && docker run --name \<Container-Name> -it \<Image-Name>

docker rm recommender && docker run --name recommender -it recommender

# Usage in recommender

docker build -t recommender .

## With volume\
docker rm recommender && docker run --name recommender --gpus all -it -v .:/workdir recommender
## Without volume
docker rm recommender && docker run --name recommender --gpus all -it recommender
