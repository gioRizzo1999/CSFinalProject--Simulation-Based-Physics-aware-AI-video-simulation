
## Setup
The project uses a conda environment.
Run these commands in a terminal:
conda env create -f environment.yml
conda activate sim2vid

## to compact simulation frames into video for comparison 
ffmpeg -framerate 24 -i frames/frame_%04d.png -pix_fmt yuv420p -c:v libx264 output.mp4



## introduction and motivations

The recent developments in artificial intelligence technology are allowing the generation of increasingly realistic, temporally coherent, high-quality videos from human textual prompts. This technology is revolutionary and is changing many industries and dimensions of life, from education to entertainment, art, science, and much more.
Even though these models can create coherent and highly convincing videos, they still have some specific limitations. One of these limitations is the inability to produce correct and consistent motion that abides by the laws of physics. This is because many state-of-the-art video generators work by statistical approximation of learned visual patterns. This approach seems, in principle, incompatible with producing numerically exact physics, because motion is always approximated and there is no direct encoding of physical laws in the model. Furthermore, this makes it equally difficult, if not impossible, to handle or set physical parameters such as gravity, friction, mass, or other quantities. This fact makes AI video generators limited in fields that require exact physical realism, such as scientific simulations for experiments or education.
Potential solutions are being researched and developed, and while there is progress and this will continue to improve, at the moment the options are still limited. Some approaches attempt to integrate physical laws directly into the training process, while others consist of guiding generation through the use of structural boundaries, also called control maps, that constrain the video in specific structural ways. These methods generally improve some aspects but cannot offer physical exactness.
This project aims to build an AI orchestration pipeline and offer a solution to this problem by leveraging a physics simulator and control maps, in order to inform and constrain the generation process of a video generative model. The user should be able to describe the scene with a prompt, specify important physical parameters, and obtain a physically correct generated scene.
The scope of this project is limited to short, simple scenes and aims to demonstrate the utility of this physics-simulated control-maps approach in the context of educational simulations and physical experiments, where precision and motion control are fundamental.
This project is based on template 4.1: Orchestrating AI Models to Achieve a Goal.
