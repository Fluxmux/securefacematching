# Secure Face Matching
#### To read my thesis download this [pdf](https://github.com/Fluxmux/master-thesis/blob/master/thesis.pdf)
We are affected with machine learning in many aspects of our daily lives, applications ranges from facial recognition to enhanced healthcare to self-driving cars. As companies outsource image classification tasks to cloud computing service providers, we see a rise in privacy concerns for both the users wishing to keep their data confidential, as for the company wishing to keep their classifier obfuscated.

In this thesis we study the applicability of [secure multiparty computation](https://en.wikipedia.org/wiki/Secure_multi-party_computation "MPC") protocols on deep learning-based face matching and try to implement a privacy-preserving face matching algorithm.

![Workflow](https://github.com/Fluxmux/master-thesis/blob/master/fig/workflow.png)

The workflow commences by acquiring an image of the clients face, this can be done by taking a photograph with the front-facing camera of the clients smartphone. The client then performs secret sharing on the image and sends the shared secret to the participating parties. The parties receive their shares and jointly compute the output of the face recognition model on the given shared image of the face.

Following software was used in this project:
* [Docker Desktop](https://www.docker.com/products/docker-desktop)
* [MongoDB](https://mongodb.com)
* [MPyC](https://github.com/lschoe/mpyc)
* [Pytorch](https://pytorch.org/)


## Installation
### Make sure following dependencies are installed
To install Pytorch go to their website ([pytorch.com](https://pytorch.org/))

To install MPyC (the secure multiparty computation framework):

```
pip install mpyc
```

## Run demo
- First of all make sure the correct hostnames and ports are set in the environment files and the config files.
- Launch the 3 Docker containers by running the "build_and_launch_servers.bat" files.
- Execute a python script from the "device/mpyc/demo" folder.
- The Docker containers should begin computing and some time later an output should be returned to the devices prompt.
- If no output is returned or a status other than 200 is returned the application has crashed and you should try to set the allocated memory for the Docker containers higher (default is too low for bigger demos).

## Commands
Launch the 3 mpc parties:
```
server/build_and_launch_servers.bat
```

Execute the secure algorithm:
```
python device/mpyc/demos/cnn.py
```
