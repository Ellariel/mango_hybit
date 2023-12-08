# The mosaik power cell agents

It is a prototype of multi-agent system simulator combining [mango](https://gitlab.com/mango-agents/mango) agents and [mosaik](https://gitlab.com/mosaik) co-simulaion platform. It was inspired by [mosaik-mango-demo](https://gitlab.com/mosaik/examples/mosaik-mango-demo), we also integrated a wind power simulator.

## Content

* The mosaik scenario explained
* Installation and execution
* The multi-agent system explained

## The Scenario
The simulation scenario consists of four components:

* A power grid that is modeled with pandapower, and a wind power simulator
* A multi-agent system with one agent for each simulated entity and one central controller agent

In this scenario the cell agents observe the power output of their entities such as generator, load, external grid etc. A controller agent regularly collects the recent feed-in from the cell agents and make instructions.The instructions are then passed back to the cell agents. For simplicity, mosaik synchronously updates the agents with data from their associated entities.

## Installation and execution
Install all requirements:

`$ pip install -r requirements.txt`

Run the simulation by executing:

`$ python scenario.py`

The output should look like this:
>Starting "WecsSim" as "WecsSim-0" ...  
Starting "Grid" as "Grid-0" ...  
Starting "MAS" as "MAS-0" ...  
...
Starting simulation.  
Simulation finished successfully.  

## The multi-agent system explained
The multi-agent system (MAS) is located in `mosaik_agents.py`. It contains the entry point for starting the MAS. Apart from the class `MosaikAgents` which implements [mosaik-high-level-api](https://mosaik.readthedocs.io/en/latest/mosaik-api/high-level.html), there is also a class `MosaikAgent` which supports the communication between mosaik and MAS itself.

The MAS consists of multiple `CellAgent`s (one for each simulated entity) and a central `ControllerAgent`. For simplicity, all agents run in the same mango container. 

The following diagram describes the message exchange between the agents during every mosaik step.

![](misc/mas.png)
