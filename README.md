# Scalable multi-agent system framework for power cell co-simulation (MASSCA)

It is a prototype of multi-agent system (MAS) simulator for the cellular approach combining *[mango](https://mango-agents.readthedocs.io/en/latest/)* agents and *[mosaik](https://mosaik.readthedocs.io/en/latest/)* co-simulaion framework for the sake of implementing the power cell model. It was inspired by [mosaik-mango-demo](https://gitlab.com/mosaik/examples/mosaik-mango-demo), we also integrated a PV and wind power simulators that are available in *mosaik* [ecosystem](https://gitlab.com/mosaik/components).

## Content

* The power cell concept explained
* The multi-agent system architecture explained
* How to use
* How to cite
* The example scenario and scalability experiments
* Sources and references

## The cell concept

In implementing the cell model and its communication layer, we followed the general concept of an energy/power cell, which is described in:
* [https://doi.org/10.1049/iet-gtd.2019.0991] A smart power cell (SPC) is a grid subsection composed of a set of power conversion units interconnected by a 3-phase AC-grid. The monitoring and control system can supervise the behavior of SPCs only taking aggregated information regarding their interfaces into account and issue aggregated control instructions when the behavior of a particular SPC at a particular transmission network bus needs to be adjusted to influence the operating state of the transmission network.
The main internal objectives: to supervise the voltage profiles, loading limits, local stability limits, to optimize the operation of the SPC according to technical and economic criteria.
The main external objectives: to coordinate the dynamic behavior in a supportive way, the flexibility to deliberately influence the active and reactive power exchange with the transmission network, to estimate and provide forecasts of the available flexibility, to adjust the active and reactive power flow exchanged with the transmission network etc.

* [https://doi.org/10.1186/s42162-022-00243-2] An energy cell (EC) encapsulates geographically or structurally coherent parts of an energy system of any scale. In EC there are a producer, a consumer, and a storage component which are connected to a controller agent which can exchange energy and information. 
The controller might be local which is localized on the top of some cell component or can be hierarchical that is considered as an agent outside certain cells and able to control a number of cells. The goal of an EC (and a controller) is to fulfill its own needs but is furthermore able to exchange energy with connected ECs.

* [[VDE](https://www.vde.com/resource/blob/1884494/98f96973fcdba70777654d0f40c179e5/studie---zellulares-energiesystem-data.pdf)] An energy cell (EC) is a kind of logically isolated energy infrastructure including all equipment used for energy conversion, transportation, distribution and storage. ECs are used to build an energy system of any scope and the cell structure is repeated at all levels of the energy system. Neighboring cells can be arranged hierarchically, so there are cells at the same level as well as superimposed and subordinate levels. 
The control of a power cell includes all the technological control equipment, including the necessary communication technologies. Control is generally performed to maintain the operational state of the network (network protection, voltage regulation, etc.) as well as to maintain the energy balance. This can, for example, be done periodically or dynamically and result in three states: balanced, oversupplied or undersupplied across all available forms of energy. The cell control is also responsible for implementing measures to prevent network outages, as well as for carrying out necessary load reductions and energy reductions, and for fault-related switching.

## The multi-agent system architecture

The multi-agent system simulator (MASS) is located in `mosaik_agents.py`. It contains the entry point for starting the MAS. Apart from the class `MosaikAgents` which implements [mosaik-high-level-api](https://mosaik.readthedocs.io/en/latest/mosaik-api/high-level.html), there is also a class `MosaikAgent` which supports the communication between *mosaik* and MASS itself.

MASS in the general case consists of several `Agent`'s, one for each modeled entity (power unit) and one for each modeled power cell. For simplicity, all agents run in the same *mango* container. 

The following diagram describes the message exchange between the agents during every *mosaik* step.

![](misc/mas.msg.drawio.png)

The following diagram describes the core flow and the related loops.

![](misc/mas.flow.drawio.png)

## How to use
### Installation

Install directly from PyPi:

`$ pip install massca`

or from cloned repository folder:

`$ pip install .`

### Usage

Import modules and specify simulators configurations within your scenario script:
```
SIM_CONFIG = {
    'Grid': {
         'python': 'mosaik_components.pandapower:Simulator'
    },
    'MAS': {
        'python': 'mosaik_components.mas:MosaikAgents'
    },
...
}

# Multi-agent system (MAS) configuration
# Demo user-defined methods are specified in methods.py to make scenario cleaner
# See MAS_DEFAULT_CONFIG in utils.py 
MAS_CONFIG = {
    'verbose': args.verbose, # 0 - no messages, 1 - basic agent comminication, 2 - full
    ...
}
```

Initialize the pandapower network and MAS simulator:
```
    gridsim = world.start('Grid', step_size=STEP_SIZE)
    massim = world.start('MAS', step_size=STEP_SIZE, **MAS_CONFIG)     
```

Instantiate model entities and agents:
```
    grid = gridsim.Grid(json=GRID_FILE)
    mosaik_agent = massim.MosaikAgents() # core agent for the mosaik communication
    
    cell_controllers = [] # top-level agents that are named as cell_controllers, one per cell
    cell_controllers += massim.MosaikAgents.create(num=1, controller=None)

    agents = [] # one simple agent per unit (sgen, load)
    agents += massim.MosaikAgents.create(num=1, controller=cell_controllers[-1].eid)
```

Connect and run:
```
    world.connect(sim, agents[-1], ('P[MW]', 'current'))
    world.connect(agents[-1], sim, 'scale_factor', weak=True)

    world.run(until=END, print_progress=True)
```

## How to cite

#### BibTeX format

```tex
@software{TODO,
author = {TODO, TODO},
license = {Apache-2.0},
month = {10},
title = {{TODO}},
url = {https://github.com/TODO},
version = {0.1.0},
year = {2024}
}
```

#### APA-like format

```
TODO, R. (2024). TODO (Version 1.0.0) [Computer software]. https://github.com/TODO
```

## The example scenario and scalability experiments
### Scenario

A simple simulation scenario consists of two components:

* A power network that is modeled with *pandapower* and a set of PV/Wind power simulators that are linked to generators in the network as well as flxible load simulators. 
* A multi-agent system with one (intra-cell) agent for each simulated entity, a number of (top-level) agents representing the aggregate (cellular) level, and one core *mosaik agent* serving the *mosaik* interface.

In this scenario agents observe the power output and flexibility provided by associated entities such as generator, load etc. Cell agents aggregate the output from connected agents and then pass it among themselves to develop internal instructions. The mosaik agent provides updates from entities to its associated agents, triggers a communication cycle, and broadcasts instructions. For simplicity, *mosaik* synchronously updates the agents with data from associated entities.

### Time complexity note

Since for demonstration purpose the aggregation function at each hierarchical level is simply the summation of flexibility measurements that are passed as agent states, and the scenario itself has no recursion (or at least no somehow increasing recursion depending on the number of cells), the time complexity must be *O(n)*. In the case of a deep hierarchy, the complexity will be as Breadth First Search *O(b^h)*, where *b* is the branching factor and *h* is the depth of the hierarchy.

### Scalablility experiments

1) To run the experiments one should go to the `scalability` folder and  install the required packages:

`$ pip install -r requirements.txt`

2) Test the scenario:

`$ python scenario.py`

3) Run the experiment sequence script:

`$ python scalability.py`

or use a docker setup:

`$ docker compose up --build`

4) Check out the results in `results` folder.

**Note.** Some Linux systems may have a user or system limit on open file descriptors. The simulation may reach this limit due to file descriptors consumed by the *mango* agents scheduler. Therefore, if you encounter this error, check `$ ulimit -a` and increase it accordingly, using e.g. `$ ulimit -n 65535`.

## Sources and references

* COHDA implementation has been adapted and [forked](https://gitlab.com/mosaik/examples/mosaik-cohda) from the [ZDIN-ZLE project](https://gitlab.com/zdin-zle/models/mosaik-cohda).





