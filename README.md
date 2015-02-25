# ![CHTMGPU Logo](http://i1218.photobucket.com/albums/dd401/222464/CHTMLOGOSMALL.png)

Continuous HTM GPU
=======

Runs a continuous (not discrete) version of HTM (Hierarchical Temporal Memory, from Numenta: ) on the GPU, and uses it for reinforcement learning.
Follow updates on my blog! [https://cireneikual.wordpress.com/](https://cireneikual.wordpress.com/)

Install
-----------

ContinuousHTMGPU relies on 2 external libraries: OpenCL and SFML. SFML is used only for visualization, and can be removed if desired.

To get OpenCL, refer to your graphics hardware vendor website (for AMD and Nvidia), or CPU vendor (e.g. the Intel OpenCL SDK).
Works best with AMD cards (best OpenCL support).

To get SFML, choose a package from here: [http://www.sfml-dev.org/download/sfml/2.2/](http://www.sfml-dev.org/download/sfml/2.2/)

ContinuousHTMGPU uses CMake as the build system. You can get CMake here: [http://www.cmake.org/download/](http://www.cmake.org/download/)

Set CMake's source code directory to the ContinuousHTMGPU root directory (the one that contains the /source folder as well as a CMakeLists.txt).

Set CMake's build directory to the same directory as in the previous step. Optionally, you can also set it to a folder of your choice, but this may make browse the source more difficult if you are using Visual Studio.

Then press configure, and choose your compiler.

It will likely error. If this happens, no fear, there is a fix!

You can specify the paths where CMake looks manually. They will appear in red if they need to be set in the CMake GUI.

SFML is a bit tricky, you have to add a custom variable entry for a variable called SFML_ROOT and set it to the SFML root directory.

When eventually the configuration does not result in errors you can hit generate. This will generate files necessary for your compiler.

You should then be able to compile and execute the program. If you are using Visual Studio, you may have to set your startup project to the ERL project, and you may have to add the source files to the project.

Quick Start
-----------

If you want to use ContinuousHTMGPU in your own project without visualization, you can strip out the SFML visualizer if desired by simply removing the "vis" directory.

First, include HTMRL.h:

```cpp
#include <htm/HTMRL.h>
```

Next, you have to create the compute system. You can specify either GPU or CPU (GPU is recommended if you have it):

```cpp
sys::ComputeSystem cs;

cs.create(sys::ComputeSystem::_gpu);
```

With that created, you need to load the OpenCL program:

```cpp
sys::ComputeProgram program;

program.loadFromFile("resources/htmrl.cl", cs);
```

Then create the agent. Fill out a vector of LayerDesc objects to describe the structure of your agent, and specify the types of the inputs (input/action/unused). In the following actions nodes are selected randomly:

```cpp
htm::HTMRL agent;

std::vector<htm::HTMRL::LayerDesc> layerDescs(5);

layerDescs[0]._width = 64;
layerDescs[0]._height = 64;

layerDescs[1]._width = 44;
layerDescs[1]._height = 44;

layerDescs[2]._width = 32;
layerDescs[2]._height = 32;

layerDescs[3]._width = 20;
layerDescs[3]._height = 20;

layerDescs[4]._width = 16;
layerDescs[4]._height = 16;

std::vector<htm::HTMRL::InputType> inputTypes(64 * 64, htm::HTMRL::_state);

for (int x = 0; x < 64; x++) {
	for (int y = 32; y < 64; y++) {
		inputTypes[x + y * 64] = htm::HTMRL::_unused;
	}
}

std::uniform_int_distribution<int> actionXDist(0, 63);
std::uniform_int_distribution<int> actionYDist(33, 63);

std::vector<int> actionIndices;

for (int i = 0; i < 8; i++) {
	int x = actionXDist(generator);
	int y = actionYDist(generator);

	if (inputTypes[x + y * 64] == htm::HTMRL::_action)
		continue;

	inputTypes[x + y * 64] = htm::HTMRL::_action;

	actionIndices.push_back(x + y * 64);
}

agent.createRandom(cs, program, 64, 64, 4, layerDescs, inputTypes, -0.05f, 0.05f, -0.05f, 0.05f, generator);
``` 

Then to use the agent, call:

```cpp
agent.setInput(x, y, <value>);
```

to set the value of an input, and:

```cpp
agent.getOutput(actionIndices[i]); // actionIndices[i] is the index of the output, from the above example
```

to get a output.

Step the simulation like this:

```cpp
agent.step(cs, reward, 0.01f, 0.01f, 0.01f, 0.05f, 0.01f, 0.05f, 0.2f, 0.5f, 0.5f, 0.5f, 0.01f, 0.2f, 0.992f, 0.15f, 0.15f, 120, 10, 2, generator);
```

The parameters above are suggested values.

Visualization
-----------

Instructions coming soon! For now just take a look at the example code, Main.cpp.

License
-----------

ContinuousHTMGPU
Copyright (C) 2014-2015 Eric Laukien

This software is provided 'as-is', without any express or implied
warranty.  In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
	claim that you wrote the original software. If you use this software
	in a product, an acknowledgment in the product documentation would be
	appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
	misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.

------------------------------------------------------------------------------

ContinuousHTMGPU uses the following external libraries:

SFML - source code is licensed under the zlib/png license.
OpenCL

