<h2 align="center">Omicron: A multi-modal agent development framework</h2>

Omicron is an agent framework for developing multi-modal agents. It is made up of a core agent object, with an agenda, memory, and a set of core mechanisms for agenda processing and memory parsing. All other agent elements are developed as mechanisms.

![agent architecture](https://raw.githubusercontent.com/omicron-ai/omicron/master/resources/images/architecture/agentarch.png)

<h3>Mechanisms</h3>

Omicron is intentionally very modular. Nearly every agent element is a modular mechanism, even fundamental things like perceptors for signal input and attention. Every agent function inherits from a base Mechanism class and have similarities, but is ultimately grouped under one of three categories: InputMechanism, OutputMechanism, or InternalMechanism.

The agents agenda is processed by an AttentionMechanism, which is a scheduler algorithm that assigns signals to their respective Mechanisms.

The agents memory is processed by a KnowledgeMechanism, which is a simple reasoner that searches, parses, and adds to the agent memory. 

<h3>Memory</h3>

The agents memory is (for now) composed of two primary spaces: Semantic and Situational. 

![agent memory](https://raw.githubusercontent.com/omicron-ai/omicron/master/resources/images/architecture/memory.png)

Semantic memory is the top level knowledge class, under which all  is stored and maintained. 

Situational memory is the top level context class, under which all contextual information is stored and maintianed. 



## Contributing

Please see [Contributing to Omicron](CONTRIBUTING.md) for information.