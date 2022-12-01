import numpy as np
import zora
import pandas as pd

def run_model(
    Generator, 
    eps=1e-5,
    sweep_cut=True,
    window=10,
    merge=True,
    merge_size_factor=200,
    merge_overlap_rate=0.8,
    search_limit=1<<24, 
    debug=False
):
    node_dataframe, edge_dataframe, node_lookup, edge_node_from, edge_node_to, \
    node_edge_begin, node_edge_end = convert_to_zora_format(Generator, debug=False)
    print(type(node_edge_begin))
    print(type(node_edge_end))
    
    node_is_reported = node_dataframe["is_reported"]
    node_retain = node_dataframe["retain"].to_numpy()
    seed_n = np.count_nonzero(node_is_reported == 1)
    seed_node = node_dataframe.index.to_numpy()[node_is_reported == 1]
    seed_rank = np.ones(seed_n)
    seed_group = np.arange(seed_n)
    node_weight = node_dataframe["weight"].to_numpy()
    edge_weight = edge_dataframe["weight"].to_numpy()
    node_cust_id = node_dataframe.index
    
    # algorithm preparation
    members = []
    queue = []
    
    def run_lcd(group):
        member_n, member_node, member_rank = zora.personalized_page_rank(
            node_edge_begin,
            node_edge_end,
            node_weight,
            node_retain,
            edge_node_to,
            edge_weight,
            0,
            np.count_nonzero(seed_group == group),
            seed_node[seed_group == group],
            seed_rank[seed_group == group],
            eps,
            search_limit
        )
        sorted_n = member_n
        sorted_member, = zora.sort(
            sorted_n,
            member_rank
        )
        if sweep_cut:
            sorted_cut, = zora.sweep_cut(
                0,
                sorted_n,
                sorted_member,
                member_node,
                node_edge_begin,
                node_edge_end,
                node_weight,
                edge_node_to,
                edge_weight,
                window
            )
        else:
            sorted_cut = 0
        members.append(
            (
                sorted_n - sorted_cut,
                member_node[sorted_member[sorted_cut:]],
                member_rank[sorted_member[sorted_cut:]],
            )
        )  
    
    def evaluate_similarity(group_1, group_2):
        member_n_1, member_node_1, member_rank_1 = members[group_1]
        member_n_2, member_node_2, member_rank_2 = members[group_2]
        if member_n_1 + member_n_2 <= merge_size_factor:
            common_rank_1 = member_rank_1 * np.in1d(member_node_1, member_node_2)
            common_rank_2 = member_rank_2 * np.in1d(member_node_2, member_node_1)
            overlap = max(
                common_rank_1.sum() / member_rank_1.sum(),
                common_rank_2.sum() / member_rank_2.sum()
            )
            score = overlap * merge_size_factor / (member_n_1 + member_n_2)
            if score >= 1 or overlap >= merge_overlap_rate:
                return score
    
    def find_merge_pairs(group):
        _, member_node, _ = members[group]
        merge_group = seed_group[np.in1d(seed_node, member_node)]
        for group_2 in merge_group:
            if group != group_2:
                score = evaluate_similarity(group, group_2)
                if score is not None:
                    heapq.heappush(queue, (-score, group, group_2))
    
    # single seed lcd
    print("lcd step 1 single-seed lcd")
    for group in range(seed_n):
        run_lcd(group)
    for group in range(seed_n):
        find_merge_pairs(group)

    # seed merging & multi-seed lcd
    if merge:
        print("lcd step 2 seed merging & multi-seed lcd")
        while queue:
            _, group_1, group_2 = heapq.heappop(queue)
            if (seed_group == group_1).any() and (seed_group == group_2).any():
                group = len(members)
                seed_group[seed_group == group_1] = group
                seed_group[seed_group == group_2] = group
                run_lcd(group)
                find_merge_pairs(group)
    
    # result collection
    print('lcd step 3 result collection')
    dataframes = []
    for group in set(seed_group):
        member_n, member_node, member_rank = members[group]
        dataframes.append(pd.DataFrame({
            'cust_id': node_cust_id[member_node],
            'group': np.repeat(group, member_n),
            'rank': member_rank,
        }))

    return pd.concat(dataframes) 


def convert_to_zora_format(Generator, debug):
    g = Generator.g
    node_n = len(g.nodes)
    print("number of nodes is: ", node_n)
    edge_n = len(g.edges)
    print("number of edges is: ", edge_n)
    index = range(node_n)
    node_dataframe = pd.DataFrame({"nodes":g.nodes},index=index)
    is_reported = []
    for i in range(node_n):
        if "is_alert" in g.nodes[i]:
            is_reported.append(int(bool(g.nodes[i]["is_alert"])))
            continue
        if "is_sar" in g.nodes[i]:
            is_reported.append(int(bool(g.nodes[i]["is_sar"])))
            continue
        else:
            is_reported.append(0)
    node_dataframe.loc[:,'is_reported'] = is_reported
    node_dataframe.loc[:,'retain'] = 0.15
    edge_weight = [np.log(g.edges[i,j,k]["amount"]) for (i,j,k) in g.edges]
    edge_dataframe = pd.DataFrame([(edge[0],edge[1]) for edge in g.edges])
    edge_dataframe.columns = ["from","to"]
    edge_dataframe["weight"] = edge_weight
    node_weight = edge_dataframe.groupby("from")["weight"].sum()
    node_dataframe = node_dataframe.join(node_weight).fillna(0)
    node_lookup = pd.DataFrame({'node': node_dataframe.index,}, index=index)
    edge_node_from = edge_dataframe.join(node_lookup, on=edge_dataframe["from"]).node.to_numpy()
    edge_node_to = edge_dataframe.join(node_lookup, on=edge_dataframe["to"]).node.to_numpy()
    node_edge_begin, node_edge_end = zora.group_detect(
        node_n,
        0,
        edge_n,
        edge_node_from
    )
    if debug==True:
        print("edge_node_from: ", edge_node_from)
        print("edge_node_to: ", edge_node_to)
        print("node_edge_begin: ", node_edge_begin)
        print("node_edge_end: ", node_edge_end)
    return node_dataframe, edge_dataframe, node_lookup, edge_node_from, \
           edge_node_to, node_edge_begin, node_edge_end
