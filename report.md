---
tags: course
---
# Link Analysis
## 2022 Data Mining Project 3 Report 
- P76114511 資訊所一 楊晴雯
- Source Code: https://github.com/Nana2929/DM-Project3 

 

## 1️⃣ Algorithm Description & Python Implementation 


Link analysis algorithms are used to rank (decide the order of display) a document / a webpage when a query is sent into a search engine, so that the users can pinpoint the pages in need faster. 

### 1. PageRank (Larry Page, Sergey Brin)
Google says that "PageRank works by counting the number and quality of links to a page to determine a rough estimate of how important the website is. The underlying assumption is that more important websites are likely to receive more links from other websites." 
The iterative version of PageRank loops through all nodes in 1 iteration to update the nodes' rank following the formula below.

$d$: damping factor 
$N$: number of pages (# of nodes)

$$ 
PR(P_i) = \frac{d}{N} + (1-d) \Sigma_{l_{j,i} \in E}PR(P_j)/OutDegree(P_j)
$$

The pagerank value of a page $i$ therefore can be seen as the probability that a random Internet surfer 🏄‍♀️  lands on page $i$ after $iteration$ times of clicks. 
The damping factor $d$ is a hyperparameter of empiricism. Consider the scenario where the surfer clicks into a sink (a page having any outbound links). And then **this surfer will randomly pick a page among all pages to restart the random walk process (with probability $d$). Otherwise the surfer follows the hyperlink structure to click on a valid outbound page (with probability $1-d$).**  
The whole process, if without number of iterations specified, should be terminated when the pageranks coverge. The damping factor $d$ plays a major role in the speed of convergence; a higher $d$ will lead to faster convergence (some papers inverts the $d$ weight to let it represent the probability that a surfer will follow the outbound structure, so thier conclusion may look textually different but carry the same meaning as mine). Note that an issue (page rank sink) exists in PageRank algorithm, and could be resolved by some improvements not explored here. 

```python=
def PageRank(G:Graph, 
            max_iters:int, 
            damping_factor:float):
    """
    Args:
        G (networkx.classes.graph.Graph): 
        max_iters (int): number of iters 
        damping_factor (float): 
        The PageRank theory holds that an imaginary surfer who is randomly clicking on links will eventually stop clicking. The probability, at any step, that the person will continue is a damping factor d.
    """
    N = G.N
    if N == 0:
        raise ValueError('Empty Graph')
    PageRanksHistory = []  
    d = damping_factor
    # 1. initialization 
    PageRanks = np.full(N, 1/N)
    # 2. iteratively update the page rank 
    for iter in range(max_iters):
        # 2-1. Create to save newly-updated pageranks 
        newPageRanks = np.zeros(N)
        # 2-2. Update according to the formula 
        for i in range(N):
            for n in G.in_neighbors[i]:
                newPageRanks[i] += PageRanks[n] / len(G.out_neighbors[n])
        PageRanks =  d/N + (1-d) * newPageRanks
    # 3. Normalize (L1-norm) after all iterations finish 
    PageRanks = PageRanks / (PageRanks.sum())
    return PageRanks

```

### 2. HITS (Jon Kleinberg)

The HITS (Hyperlink-Induced Topic Search) algorithm starts with a set of pages clled **root set** that are relevant to the query, and then further expands to include the pages that out-link to these root pages. The expanded set is called **base set**. HITS only focuses on **the subgraph of the base set and the outlinked pages within base set** (in this assignment, **we assume the given graph is the focused subgraph**). 
HITS record 2 values:
- authority score: the pages containing highly relevant information
- hubness score: the pages pointing to authoritative pages 

HITS updates the 2 values in mutual recursion: 

$Hub(P_i) = \Sigma_{l_{j,i}}Auth(P_j)$ // Sum of in-neighbors's Auth
$Auth(P_i) = \Sigma_{l_{i,j}}Hub(P_j)$ // Sum of out-neighbors's Hub 

For the sake of convergence, the auth and hub scores are then normalized using L1 or L2 norm. The below implementation uses L1 as default (as the assignment requires). With this algorithm, we are able to identify good auths (relevancy) and the good hubs (endorsement) in the subgraph. 

```python=
def HITS(G:Graph, 
    max_iters:int, 
    norm = 'L1')-> Tuple[np.array, np.array]:
    """
    HITS(Hyperlink-induced topic search)
    Authority: Providing valuable infor on certain topic 
    Hub: Give good supports to those pages with high authority
    - A good hub increases the authority weight of the pages it points. 
    - A good authority increases the hub weight of the pages that point to it. 
    The idea is then to apply the two operations above alternatively until equilibrium values for the hub and authority weights are reached.
    Args:
        G (Graph): the given subgraph 
    Returns:
        Tuple(np.array, np.array): Auth, Hub Vectors 
            Auth: shape (N, ) Auth[n] is the authority score of node n
            Hub: shape (N, )  Similarly, Hub[n] is the hub score of node n
    """
    auths = np.ones(G.N)
    hubs = np.ones(G.N) 
    def get_update_Auth(n):
        # authority: the nodes being highly pointed to 
        return hubs[G.in_neighbors[n]].sum()
    def get_update_Hub(n):
        # hub: the nodes pointing to others 
        return auths[G.out_neighbors[n]].sum()
    
    for _ in range(max_iters):
        new_auths = np.zeros_like(auths)
        new_hubs = np.zeros_like(hubs)
        for n in range(G.N):
            new_auths[n] = get_update_Auth(n)
            new_hubs[n] = get_update_Hub(n)
        if norm == 'L1':
            auths = new_auths / np.sum(new_auths)
            hubs = new_hubs / np.sum(new_hubs)
        else: 
            # wiki: L2 norm 
            # https://en.wikipedia.org/wiki/HITS_algorithm
            auths = new_auths / np.sqrt(np.sum(new_auths**2))
            hubs = new_hubs / np.sqrt(np.sum(new_hubs**2))
    
    return auths, hubs 
```


### 3. SimRank (Jeh, Widom)

Simrank is used to measure the topological similarity of 2 objects by objects that are referencing both of them. Like the above 2 algorithms, it updates values iteratively (using the previous iteration's numbers); but unlike the above 2, it updates values by pair of nodes. The formula is defined as follows, where $I(*)$ is the in-neighbors of node node *. For any node without in-neighbors, its associated SimRank is 0. For pair of identical nodes, their SimRank is 1. SimRank is used in web aplication for finding similar-content documents or to cluster webpages, or for recommender system to group similar-preference users. 

$$
SimRank(a, b) = \frac{C}{I(a)I(b)}\Sigma_{i=1}^{|I(a)|}\Sigma_{j=1}^{|I(b)|}SimRank(I_i(a), I_j(b))
$$



```python=
@timer 
def SimRank(G: Graph, 
            max_iters:int, 
            decay_factor:float):
    # SimRank_sum = the sum of SimRank value of all in-neighbor pairs (SimRank value is from the previous iteration)
    C = decay_factor 
    def get_update_simrank(
                    a:int, 
                    b:int, 
                    simRank: np.array):
        if a == b: 
            return 1    
        a_in_neighbors = G.in_neighbors[a] # I_i(a)
        b_in_neighbors = G.in_neighbors[b] # I_j(b)
        a_in_size, b_in_size = len(a_in_neighbors), len(b_in_neighbors)
        if not a_in_size or not b_in_size:
            return 0
        temp = 0 
        for i in a_in_neighbors:
            for j in b_in_neighbors:
                temp += simRank[i, j]
        # scaling the simRank 
        return C * temp / (a_in_size * b_in_size) 
                        
    simRank = np.zeros((G.N, G.N))
    for iter in range(max_iters):
        newSimRank = np.zeros_like(simRank)
        for a in range(G.N):
            for b in range(a, G.N):
                newSimRank[a, b] = newSimRank[b, a] = get_update_simrank(a, b, simRank)
        simRank = newSimRank.copy() 
    return simRank    

```

## 2️⃣ Result Analysis & Discussion 


### graph_1.txt 
![](https://i.imgur.com/ESbtkH7.png =300x)
```
PageRank 
0.056 0.107 0.152 0.193 0.230 0.263

Hub 
0.200 0.200 0.200 0.200 0.200 0.000

Auth 
0.000 0.200 0.200 0.200 0.200 0.200

SimRank
1.000 0.000 0.000 0.000 0.000 0.000
0.000 1.000 0.000 0.000 0.000 0.000
0.000 0.000 1.000 0.000 0.000 0.000
0.000 0.000 0.000 1.000 0.000 0.000
0.000 0.000 0.000 0.000 1.000 0.000
0.000 0.000 0.000 0.000 0.000 1.000
```
Graph 1 shows an increasing pagerank; page 1 has no in-links so it is ranked the minimum, page 6 accumulates from all the other pages so it is ranked the maximum. For the hub values, page 6 endorses no other pages so it is 0 while others evenly distribute the hub scores. For the auth values it is the reverse. For SimRank, no 2 nodes outlink to 1 same node, so the similarity score is 1 for the diagonal entries and 0 otherwise. 

---

### graph_2.txt
![](https://i.imgur.com/VsuVNhu.png =300x)

```
PageRank
0.200 0.200 0.200 0.200 0.200 

Hub
0.200 0.200 0.200 0.200 0.200 

Auth
0.200 0.200 0.200 0.200 0.200 

SimRank
1.000 0.000 0.000 0.000 0.000
0.000 1.000 0.000 0.000 0.000
0.000 0.000 1.000 0.000 0.000
0.000 0.000 0.000 1.000 0.000
0.000 0.000 0.000 0.000 1.000
```

Graph 2 is a circular graph with 1 in- and out- link per page, and therefore the pagerank that representing the probability of landing on any page is equal -> 0.2. Same reasoning holds for the Auth and Hub values. As for the SimRank matrix, there are no 2 nodes out-linking to the same node so the result is an identity matrix as graph 1's. 

---

### graph_3.txt 
![](https://i.imgur.com/lku9pRu.png =300x)

```
PageRank
0.172 0.328 0.328 0.172 

Hub
0.191 0.309 0.309 0.191 

Auth
0.191 0.309 0.309 0.191 

SimRank
1.000 0.000 0.538 0.000
0.000 1.000 0.000 0.538
0.538 0.000 1.000 0.000
0.000 0.538 0.000 1.000
```
Pagerank and Auth ranks node 2, 3 higher because they are endorsed more (2 inlinks). Hub ranks 2, 3 higher too because they endorse others more too (2 outlinks).as for SimRank, they are supported by 2 pages simultaneously, so pages that link to 2 (1, 3) are similar (sim(1,3) = 0.538). Same for the pages linking to node 3 simultaneously (2,4). 


### damping factor $d$ in PageRank 

```
# Experimented on graph_3.txt 
# under 30 iterations 
d = 0.1 [0.172 0.328 0.328 0.172]
d = 0.3 [0.185 0.315 0.315 0.185]
d = 0.6 [0.208 0.292 0.292 0.208]
```
It is observed that as $d$ goes towards 1 (probability of random restart), the pageranks of all pages are closed to be equal. This is reasonable because random restarts do not capture the huperlink structure of the graph and approximates the scenario where pages outlink to all other pages. Setting $d$ in $(0.1, 0.15)$ seems to be a good idea based on the probability of a random click for a user in empirical world. Google uses 0.15 (0.85 in the inverted formula) in their search engine. 

### decay factor $c$ in SimRank 

```
# Experimented on graph_3.txt
# under 30 iterations 

C = 0.6 
[[1.    0.    0.429 0.   ]
 [0.    1.    0.    0.429]
 [0.429 0.    1.    0.   ]
 [0.    0.429 0.    1.   ]]
 
C = 0.7 
 [[1.    0.    0.538 0.   ]
 [0.    1.    0.    0.538]
 [0.538 0.    1.    0.   ]
 [0.    0.538 0.    1.   ]]
 
C = 0.8
 [[1.    0.    0.667 0.   ]
 [0.    1.    0.    0.667]
 [0.667 0.    1.    0.   ]
 [0.    0.667 0.    1.   ]]
 
C = 1
 [[1. 0. 1. 0.]
 [0. 1. 0. 1.]
 [1. 0. 1. 0.]
 [0. 1. 0. 1.]]
```

It is observed that $c$ has the same tendency as $d$; a higher $c$ (close to 1) makes pages with same number of incoming links have similar scores (setting $c$ to 1 makes $SimRank(2,3)=1$), and the while process converges faster. Empirically $c$ is set in $(0.6, 0.8)$. 


## 3️⃣ Find A Way 



### Revised graph_1.txt
![](https://i.imgur.com/8UzIeIM.png =300x)

<font color="#447CB5"> Add a new node 7 and a new link (7,1)</font>, so that 1 is promoted to an authoritative status by 7 while keeping its hubness not too low. Note that adding another node 8 and a link (8,1) will result in hubness of 1 to 6 dropping to 0 (because node 7, 8 share the hubness). **Since there's a trade-off between hubness and authorities, it is impossible to improve on both values without altering the structure greatly.** 

```
# Before 
PageRank 
0.056 0.107 0.152 0.193 0.230 0.263

Hub 
0.200 0.200 0.200 0.200 0.200 0.000

Auth 
0.000 0.200 0.200 0.200 0.200 0.200


# After revision 
Pagerank
0.167 0.167 0.167 0.167 0.167 0.167 0.000 

Auth
0.167 0.167 0.167 0.167 0.167 0.000 0.167 

Hub
0.082 0.118 0.149 0.178 0.203 0.226 0.043 
```


---

### Revised graph_2.txt

![](https://i.imgur.com/qKEXgZQ.png =300x)

<font color="#447CB5"> Add the edges (2,1), (3,1), (1,3), (1,4).
</font> Node 1 now has 3 in-links and 3 out-links, in which 2 edges [1,2], [1,3] are bi-directed, meaning that these 2 pairs endorse each other. Some of node 1's in-links (2,5) have nice hubness, hence the improved authority value; all of the out-links (2,3,4) similarly have good authority values, hence the improved hubness value. 

```
# Before 
PageRank
0.200 0.200 0.200 0.200 0.200 

Hub
0.200 0.200 0.200 0.200 0.200 

Auth
0.200 0.200 0.200 0.200 0.200 

# After revision 
PageRank
0.324 0.117 0.170 0.194 0.194 

Hub
0.315 0.270 0.270 0.000 0.145 

Auth
0.315 0.145 0.270 0.270 0.000 
```


---

### Revised graph_3.txt

![](https://i.imgur.com/xgqGzn9.png =300x)
Following the structure, <font color="#447CB5"> I add (1,5)</font> and 3 values are promoted.




```
# Before
PageRank
0.172 0.328 0.328 0.172 

Hub
0.191 0.309 0.309 0.191 

Auth
0.191 0.309 0.309 0.191

# After revision 
Pagerank 
 0.244 0.236 0.244 0.138 0.138
Auth
 0.214 0.286 0.214 0.143 0.143
Hub
 0.214 0.286 0.214 0.143 0.143
```


## 4️⃣ Efficiency Analysis

- Testing Environment: 
    - Hardware: MacBook Pro 2020 
       CPU: Intel(R) Core(TM) i5-8257U CPU @ 1.40GHz
       (Check using `sysctl -n machdep.cpu.brand_string`)
    - IDE: VSCode Version: 1.74.0

  The statitics are recorded in seconds (`s`) and rounded to 2 decimal places. 
  It could be observed that SimRank almost always takes the longest because while the other 2 caclulates 1 score for 1 node, it calculates 1 score for a pair of nodes. As the graph grows larger with more nodes and edges, SimRank is affected most dramatically, the next is HITS because it considers a node's in-links and out-links for Auth and Hub caculation; Pagerank comes the last because only number of in-links affects the runtime; the number of out-links is used only as divisor in the formula. 
  

    - 30 iterations 
        | Graph | (#nodes, #edges)| PageRank | HITS | SimRank |
        | -------- | ----|-------- | -------- | -------- |
        | graph_1     |(6,5)| 0     | 0.01    |0    |
        | graph_2     |(5,5)| 0    | 0     |0     |
        | graph_3     |(4,6)|0     | 0     |0     |
        | graph_4     | (7,18)|0     | 0    |0   |
        | graph_5     | (469, 1102)|0.02     | 0.12    |6.26     |
        | graph_6     | (1228, 5220)|  0.1 |  0.46   |  135.35  |
        | ibm-5000    |(836, 4798) |   0.09 | 0.3  |  107.22  |

    - 100 iterations 
        | Graph | (#nodes, #edges)| PageRank | HITS | SimRank |
        | -------- | ----|-------- | -------- | -------- |
        | graph_1     |(6,5)| 0     | 0.02    |0.01     |
        | graph_2     |(5,5)| 0    | 0.01     |0.01     |
        | graph_3     |(4,6)|0     | 0.02     |0     |
        | graph_4     | (7,18)|0     | 0.01    |0.01     |
        | graph_5     | (469, 1102)|0.11     | 0.65    |36.97     |
        | graph_6     | (1228, 5220)|0.32   | 1.54     |531.04     |
        | ibm-5000    |(836, 4798) | 0.31   | 1.10     |  282.12  |

## 5️⃣ References

1. [Wiki: PageRank](https://en.wikipedia.org/wiki/PageRank#:~:text=from%20page%20v.-,Damping%20factor,is%20a%20damping%20factor%20d.) 
2. [Atul Kumar Srivastava et al, Discussion on Damping Factor Value in PageRank Computation](https://www.mecs-press.org/ijisa/ijisa-v9-n9/IJISA-V9-N9-3.pdf) 
3. [PageRank course slides from Brown University](https://cs.brown.edu/courses/cs016/static/files/assignments/projects/GraphHelpSession.pdf) 
4. [Youtube: HITS Algorithm Explained by Mohit](https://www.youtube.com/watch?v=-kiKUYM9Qq8)
5. [阻尼因子（damping factor）對網頁排名之敏感度分析](https://www.stat.purdue.edu/~dkjlin/documents/publications/2005/2005_JCSA.pdf)
6. [chonyy's Medium posts](https://towardsdatascience.com/simrank-similarity-analysis-1d8d5a18766a)
7. [C in SimRank](https://pdf.sciencedirectassets.com/271992/1-s2.0-S0168927420X00031/1-s2.0-S0168927420300453/am.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEEsaCXVzLWVhc3QtMSJGMEQCIAys%2BEBmrypJgCWuXMVlNp2Vq%2FNu6J40ilHR4WB0PIj1AiA2jiI1d8MkvEhu%2F45fo%2B%2BaUk8SKSYLeeoQOPrbeoDb%2ByrVBAiE%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAUaDDA1OTAwMzU0Njg2NSIMn8Oq0jk6LgRHLMaqKqkEUyIh5wkkFIeJ4OAxQo3PgwLWcF27AqBwYsTHBR%2BBerD%2BlxQ5a%2BSA6w9pxVVR%2B2oTIptB%2BliVy%2B9SsXaX3aK9LxwhfhqYQ32GZcro29TwlS%2BOpjNqQm8faZYGRqy1pRH55HmK5tnxmTWbIEFcq8bbqWZ0L0rLv3znhJA9rauhvC%2Fvcvs%2FCxa3lWbQ6WwgeHDi%2BjoQ%2BwigzvrU8qz%2BLjENpBF0dAS3Na8pkg1%2FIpHNBTM2rRxJf8Fj6xy6P4z16xF1NJ3s3VoIcVNHcnNNYCXQpTXYCwyMUy99b48jSIgsKlCpohAsO%2FiWAdoS2rWNsdMaSo0uxg9hhckSqD85oEeXeqpNq6qWV2M%2BH5yPDtsA0%2BBk48Te9k1pOkaxukxgLrqUOZmqpBD1d0KhM9rmpcb6mEDu2WV2s7XtHwCs2SmoMPEN3U2qPxvDrB1S2Hb9hra3e8%2BkWLeA1uw6dheAUv85xIgDp2PKBdxLuh4DTp8w7YU5ENk0c7eaJkj1jijmRJNxJuZU5TUlKEHNBk7kTXn0kedTOVIQibMiYbpGKPl3QKBZpgb8c42EGfWnf0ui2QK2HI5sys994DXR5ML79%2BL8GIBQlCqiHiqGKp635KqnlD4vOT%2BgShvcnXtgiNtb7GuiXyfMPOpY29H49UKtAgaUFHs%2BGd5q6mW1NzCXx6QhxdbPTtZPXEGTSRjaR50ruUCTtr3%2BZmdTcx7uocjNIToLSfv8pTefvfLVzjCm44mdBjqqAfecKLHO3GjWd6FFz%2B9bUGLdhMFbDTnFItXTI%2F%2FXQu0mfZYK9ZcuVBrSbDbXb5MP4gR5x82UxZznSYr0v7IOpQTmasbLg5Tar4L%2B70qTq%2B8Xi4wkDqFMcaPuzdK08k7RUxMRTyNuagDuPf4ml4Uswvz2mvDpcNxY9R5xk3VQOI8N3PeL4IbhEBM39TbnxlkLGjrhPSGOz%2FEIYUPKXO5Rrr8Am2E1gSMiCkjq&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20221221T030020Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY6WLIOJUO%2F20221221%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=9f69b397a400f526c3f6f62502b7c347236fef78408327eadb5190524b799302&hash=0bc6b83ea6e3e84bcc4059cc2286ba89ae89eb6b0ba788500b05000b9ccb7e0d&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0168927420300453&tid=pdf-2b6a14ff-19f1-4ddd-8742-c279780ae884&sid=a5a7a2bd61f955423a7b2e3196f3ecc77cdbgxrqa&type=client)