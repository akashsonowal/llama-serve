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

### Other steps:

change hostname
```
sudo hostnamectl set-hostname k8s-control-plane
```

add ip of all nodes in same subnet
```
sudo nano /etc/hosts 
```
add extra kernels in conf
```
cat <<EOF | sudo tee /etc/modules-load.d/containerd.conf
> overlay
> br_netfilter
> EOF
```
manually apply the kernel modules
```
sudo modprobe overlay
sudo modprobe br_netfilter
```
setup networks
```
cat <<EOF | sudo tee /etc/sysctl.d/99-kubernetes-cri.conf
net.bridge.bridge-nf-call-iptables = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward = 1
EOF
```
to apply them
```
sudo sysctl --system
```
```
sudo apt-get install gnupg
```
install docker and containerd
```
sudo containerd config default | sudo tee /etc/containerd/config.toml
```
```
sudo swapoff -a
```
```
sudo systemctl restart containerd
sudo systemctl status containerd
```
```
follow the kubernetes installation guide https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/install-kubeadm/#installing-kubeadm-kubelet-and-kubectl



### Reference

https://www.youtube.com/watch?v=xX52dc3u2HU