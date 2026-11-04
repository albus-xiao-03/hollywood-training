# Hollywood Training

Visualize PyTorch network training process with a Hollywood-style GUI.

---

## Objective
Build a Python package `hollywood-training` to visualize PyTorch network training process with a Hollywood-style GUI.

---

## Requirements
1. The package **MUST** provide a method to automatically track the architecture of the defined network, including the weights of each layers.
    The method **CAN** be **ANY** Python objects or mechanisms, but a `decorator` applys to the class of the network or a `slot dataclass` designed to contain the network including the linear components and activation functions may be preferred.
2. The package **MUST** provide a method to automatically manage the training process. 
    It **MUST** allow users to defind the training process in the native PyTorch-style way. Then, wrap the defined training process into a Python object. And, pass the wrap object to the management method the package provides. 
    The package may provide several `decorator` or `class` or other similar Python object to distinguish the different steps or components of the training process. 
3. **BOTH** requirements `1` and `2` **MUST NOT** significantly change the native PyTorch workflow. 
4. The package **MUST** provide a native GUI window, **NOT** a Web page.
    The GUI **MUST** provide smooth animations to demonstrate the **REAL** training process.
5. **ALL** the training update **MUST** be synchronized with the GUI update. 
    The GUI **MUST** be updated at **EVERY** `epoch`, and complete update **BEFORE** the next `epoch`.
    Computational efficiency has **NO** priority in this project. 
6. The training process **MUST** run as fast as synchronization allows. **DO NOT** consider the readability of the GUI animations.
7. The package **MUST** provide a set of well-defined commands to control the training process managed by the package during runtime.
8. **ALL** `dark magic` Python methods are **STRICTLY PROHIBITED**.
    Example of `dark magic`: Dynamically retrive the definition of user's training function, analyze user's intention with function definition by `regex`, then determine the necessary information of training process.
9. The package **MUST** be **COMPLETELY** decompoised into modules with clear architecture.
10. **ALL** package file **MUST** be located in the package folder `hollywood_training`.
11. The package **MUST** provide a demo with illustrative network architecture and training dataset.
    **ALL** demo related files **MUST** be located in the folder `demos`, outside of the package folder `hollywood_training`, directly under the project folder.
12. The package **MUST** have a `README.md` for quick start.
13. The package **MUST** have a `wiki` folder provides separated `markdown` files to explain **EVERY** component of the package **AND** the overall architecture. 
    The compatibility of `GitHUB wiki` and the `wiki` folder is preferred. 

---

## GUI Layout
1. The GUI **MUST** demonstrate the architecture of the network, the connection between each layer's neurons, and the **REAL** weights of each layer that should be changed with **REAL** training update along with **EACH** `epoch`.
2. The GUI **MUST** demonstrate the information of training status, which includes but not limited to:
    - `total epoch`
    - `current epoch`
    - `batch size`
    - `learning rate`
    - `loss function`
    - `optimizer`
    - `normalization strategy`
    - `progress bar`
    - `process timer`
3. The GUI **MUST** crave the loss figures **DYNAMICALLY** along with **EACH** `epoch`, including:
    - `training loss`
    - `validation loss`
4. The GUI **MUST** provide the information of current usage status, including but not limited to:
    - `memory usage`
    - `cpu usage`
    - `gpu usage` (Only when appliable)
5. The GUI **MUST** provide a integrated terminal to allow users to control the managed training process with defined commands.
6. The GUI **MUST** show the name of the training network(its class name).
7. The GUI **MUST** be blue based.

---