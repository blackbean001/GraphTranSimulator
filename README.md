Now, you can run some algorithms (such like zora lcd, EdMot...) using the simulated data.
Try main.ipynb ~~~

GraphTransSimulator
A simulator to generate transaction graph based on pre-defined degree distribution, account information, normal transaction and alert transaction information.

pipline

generate a multiedge graph that follows the in_degrees and out_degrees in degree.csv. Each node represents an account, each edge represents a transaction;
assign account attributes defined in account.csv to each node, including initial balance, bank_id, region and so on. The account.csv can be written in two different formats: detailed format and aggregated format.
assign normal patterns to all edges (transactions) following the parameters in normalPatterns.csv;
generate alert patterns to the graph following the alertPattersn.csv;
write to csv files


patterns
In the current version, the normal patterns include: fan_in, fan_out, forward, mutual, single, periodical; the alert patterns include: fan_in, fan_out, random, cycle, bipartite, stack_bipartite, gether_scatter and scatter_gether.
One can add new patterns by writing codes in the corresponding locations (generator.py for alert patterns and nominator.py for normal patterns).



generate new graph
To generate a new simulation graph, one can first change the parameters in ./paramFiles, then run main.py.
