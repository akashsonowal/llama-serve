# llama-serve
The fastest llm server that supports 1000 request/sec

## Steps

## 1. Setting up Kubernetes Cluster from scratch

For our setup we will use 3 nodes each node with 1 GPU

## 2. Setup manifest for kubernetes

## Container registry creation (Quay or Github repository)


### GPU Processes 

```
nvtop
```

```
scp -r replace_local_folder_path  mapcreation@10.168.16.157:/mounted_nfs_data/"
```
## Steps

### Install Docker Engine
```
https://docs.docker.com/engine/install/ubuntu/
```

### Install container toolkit
```
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
```

### Build docker container and run container
```
docker build -t llama_serve:v1 .
```

```
docker run --gpus all -v /home/ubuntu/llama-serve/artifacts/:/home/ubuntu/llama-serve/artifacts/ -p 8080:8080 llama_serve:v1
```

### Install Kubernetes 

1. To setup cluster we can use either kubeadm or kops
    - follow kudeadm commands
    - whitelist the traffic
2. Setup CNI plugins like Calico etc

### Reference

https://www.youtube.com/watch?v=xX52dc3u2HU