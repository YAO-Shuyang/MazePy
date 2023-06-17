# Overview of MazePy
The thriving field of ``system neuroscience`` has enhanced the status of behavioral experiments in daily research. As the direct readout of certain brain's activities, the behavior of animals has been put at the center of the stage for 100+ years. While generations of scientists have established numerous ``behavioral paradigms`` to address the functions of diverse brain circuits, but scarce packages were developed to help people process these behavioral data, which make system neuroscience research hard for people who is not good at programming to join in.
<br/>
<br/>
Of course there're several excellent Python sofewares developed to address this issue, e.g, [``DeepLabCut``](https://github.com/DeepLabCut/DeepLabCut). DeepLabCut provides a strong function in processing animals' movements collected by cameras (e.g., the moving trajectory of the limbs of rats) and is thus hotly welcomed by researchers. However, moving trajectory is only one of the attributes of animals' behavior, which means there are other things we could do with the behavioral data. So, we'd like to emphasize here that we are certainly not pursuing substituting these preexisting, well-developed, and powerful tools with our ``MazePy``.
<br/>
<br/>
### 1. Design environments with diverse shapes and inner structures
In the research field of the spatial navigation system of the brain, one common thing researchers like doing is manipulating the shape of an environment and training mice or rats to navigate or randomly forage within this environment. Tens of, if not hundreds of, environments or mazes (T maze, Y maze, 8-arm maze, linear track, L maze, 6-arm maze, pin maze, complex maze, 3D maze) have been designed in the past 50+ years to investigate the neural representation of the space. 

However, packages trying and setting up to unify the analysis of different shapes of the environment are scanty. So, one of our aims is to develop a package that could well fit the analysis of most of the environments, so that researchers would no longer need to write new code to process data collected in an entirely different environment.
<br/>
<br/>
MazePy provides a ``designer GUI`` to assist you in designing a new environment, and functions to help you analyze the data in the environment:
<br/>
<br/>
![GUI](https://github.com/YAO-Shuyang/MazePy/blob/master/behav/mazeobj/guiasset/GUI_backgound.png)

### 2. To integrate neural activity collected by calcium imaging and electrophysiological approaches.
After tens of years of rapid development, several tools have been developed to record neural activities, which could be divided, in general, into three groups:

- Genetically encoded calcium indicator (e.g., GCaMP family etc.)
- Electrodes (e.g. tetrode array, Neuropixel, etc.)
- Genetically encoded voltage indicator (GEVI)

Several useful tools have been developed to process raw neural data, e.g., [``suite2p``](https://github.com/MouseLand/suite2p), [``CNMF-E``](https://github.com/inscopix/inscopix-cnmfe) and numerous spike sorting algorithms. However, the one that make people thrashed out is, generally,  different analysis packages save the processing results in divergent data form, which makes code imcompatible to be used in dealing with another data forms. Here, we are trying to establish very basic but standardized data structure, to make analysis of different forms of data more easily.
<br/>
<br/>

# Plans
The first step focuses on the development of a subpackage of MazePy for the analysis of data in the field of spatial navigation. It would include an analysis of the animal's trajectory in an environment with different shapes. Mazepy would help people define different kinds of environments easily.

<br/>
Besides, we would also provide classes and functions to analyze calcium imaging data. And of course, a package to analyze electrode-recorded data has also been taken into consideration.

<br/>
In the future, we'd also consider such a possibility: using the AI tools to help us analyze data, plot high-qualified scientific figures, and write scripts automatically, which I believe would benefit researchers a lot. (开始画饼)

<br/>
We have to emphasize that the development of the package is still in a very preliminary state and may take years to finish.

<br/>

# Contact
We are certainly very open to new members! Welcome every talented and skillful person to join us! Here're some basic requirements for joining our group:

1. Have, at least, an 1-year experience of 'Python' programming.
2. Have interests in neuroscience and enthusiasm in working with others.

If you are interested in our project, feel free to reach out to Shuyang: qzyaoshuyang3@pku.edu.cn

