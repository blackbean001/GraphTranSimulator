This project is inspired by [AmlSim](https://github.com/IBM/AMLSim)

Now, you can run some algorithms (such like zora lcd, EdMot...) using the simulated data. 

Try main.ipynb ~~~

# GraphTransSimulator
A simulator to generate transaction graph based on pre-defined degree distribution, account information, normal transaction and alert transaction information.


# pipline
1. generate a multiedge graph that follows the in\_degrees and out\_degrees in degree.csv. Each node represents an account, each edge represents a transaction;
2. assign account attributes defined in account.csv to each node, including initial balance, bank_id, region and so on. The account.csv can be written in two different formats: detailed format and aggregated format.
3. assign normal patterns to all edges (transactions) following the parameters in normalPatterns.csv;
4. generate alert patterns to the graph following the alertPattersn.csv;
5. write to csv files

# patterns
In the current version, the normal patterns include: fan\_in, fan\_out, forward, mutual, single, periodical; the alert patterns include: fan\_in, fan_out, random, cycle, bipartite, stack\_bipartite, gether_scatter and scatter\_gether.
One can add new patterns by writing codes in the corresponding locations (generator.py for alert patterns and nominator.py for normal patterns).

![image](https://github.com/blackbean001/GraphTranSimulator/blob/main/pics/normalPatterns.png)
![image](https://github.com/blackbean001/GraphTranSimulator/blob/main/pics/alertPatterns.png)

# generate new graph
To generate a new simulation graph, one can first change the parameters in ./paramFiles, then run main.py.

The generated graph looks like the following:

![image](https://github.com/blackbean001/GraphTranSimulator/blob/main/pics/outputgraph.png)
![image](https://github.com/blackbean001/GraphTranSimulator/blob/main/pics/normalpattern.png)
![image](https://github.com/blackbean001/GraphTranSimulator/blob/main/pics/alertpattern.png)

