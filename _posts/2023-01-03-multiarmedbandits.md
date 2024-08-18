---
title: "MultiArmedBandits with Stochastic, Adversarial, and Adversarially Corrupted Stochastic Rewards"
permalink: "/MultiArmedBandits/"
layout: post
mathjax: true
---

The multi-armed bandit (MAB) is a classic problem in probability theory and statistics that models exploitation-exploration trade-off, leveraging choices that have proven effective in the past versus choosing new options that might provide better-unexplored trade-offs. Imagine a row of slot machines, each with a different and unknown distribution of paying out a reward. The goal is to select the machine(or 'arm') at each time instance such that it maximizes the total reward over time. This article will discuss simulations for some known algorithms in MAB in stochastically chosen, adversarially chosen, and stochastically chosen but strategically corrupted reward scenarios and run simulations for the algorithms discussed.




### Overview of the Algorithms:
Notations used:
+ $$K$$: number of arms/actions
+ $$[K]$$: the set of all arms/actions
+ $$a \in [K]$$: a unique arm/action
+ $$T$$: total number of rounds
+ $$t \in [T]$$: a specific round
+ $$a_t$$: arm chosen in round $$t$$
+ $$X_{a\in [K],t\in [T]}$$: reward for arm $$a$$ in round $$t$$ 
+ $$\Delta_{a,t}$$: is the difference between the reward of optimal arm and arm $$a$$ in round $$t$$.
+ $$n_t(a)$$: is the number of pulls of arm $$a$$ up to round $$t$$.
+ a*: be the optimal arm.
+ $$\mathbb{I}\{predicate\}$$: indicator function on predicate, it is $$1$$ if predicate is true 0 otherwise. 
&nbsp;
&nbsp;
&nbsp;

#### Stochastic Bandits:

Stochastic bandits are a type of problem in which the rewards of different actions are unrelated to each other, and the rewards of the same action at different time steps are an i.i.d distribution. 

| **Protocol:** |
|------------------|
| ***Parameters***: $$K$$ arms, $$T$$ rounds, $$T > K$$, for each arm $$a \in [K]$$, the reward for arm $$a$$ is drawn from distribution $$D_a$$. |
|For each round $$t \in [T]$$ the algorithm chooses an $$a_t\in[K]$$ and observes a reward $$X_{a_t,t}$$ |


The algorithm aims to minimize the deficit suffered from not always choosing the arm, with the highest total expected reward.
Let's define this deficit as Pseudo-Regret: 

$$R = \max_{a\in [k]} \mathbb{E}\Big[\sum_{i\in[T]}X_{a,t} - \sum_{i\in[T]}X_{a_t,t} \Big]$$

For this article, let's assume that all stochastic rewards are drawn from Normal Distribution with a mean between 0 and 1, 1 standard deviation, and capped between -5 and 6.

&nbsp;
&nbsp;
&nbsp;


#### Explore-then-commit:

Let's start with a simple algorithm: explore arms uniformly selecting each action $$N$$ times *(exploration phase)* and then committing to the best arm for the remaining $$T-NK$$ rounds *(exploitation phase)*.

This simple-looking algorithm suffers from sub-linear i.e. o(T) regret.

Proof sketch: 

From [Hoeffding's inequality](https://en.wikipedia.org/wiki/Hoeffding%27s_inequality), we can infer that if $$N$$ is chosen large enough, then the mean reward for each arm estimated by sampling in the exploration phase is almost equal to the true mean reward of the arm. Then the total regret for exploration rounds is bounded by $$NK$$ times $$\max_{a\in[K]}\Delta_a$$, and the total regret in exploration rounds should be close to negligible.


<p style="font-size: small; font-style: italic;">(<a href = "https://en.wikipedia.org/wiki/Hoeffding%27s_inequality">Hoeffding's inequality</a> is a type of <a href = "https://en.wikipedia.org/wiki/Concentration_inequality">Concentration inequality</a>, such inequalities come in useful for proving bounds on various bandits algorithms.)</p>

Let $$\mu_a$$ be true mean of arm $$a$$ and $$\overline{\mu}_{a,t}$$ be the mean estimated by sampling untill round $$t$$. 
Then, by [Hoeffding's inequality](https://en.wikipedia.org/wiki/Hoeffding%27s_inequality)

$$\implies\mathbb{P}\big[ \lvert \mu_a - \overline{\mu}_{a,t} \rvert > \delta_{a,t} \big] \leq  2e^{\frac{-2n_a(t)\cdot\delta_{a,t}^2}{(6 - (-5 ))^2}} = 2e^{\frac{-2n_a(t)\cdot\delta_{a,t}^2}{121}}$$

Let $$\delta_{a,t} = \sqrt{\frac{2\cdot 121\cdot log(T)}{n_a(t)}}$$

$$\implies \mathbb{P}\big[ \lvert \mu_a - \overline{\mu}_{a,t} \rvert > \delta_{a,t} \big] \leq\frac{2}{T^4}$$

By Union Bound, $$\mathbb{P}\big[ \cup_{\forall a \in [K],t\in [T]} \lvert\mu_a - \overline{\mu}_{a,t} \rvert > \delta_{a,t}\big] \leq \sum_{t\in[T]}\sum_{a\in[K]} \frac{2}{T^4} \lt \frac{2}{T^4}$$

<p style="font-size: small; font-style: italic;">(For this particular proof we don't need such a strict condition on the difference between empirical and true mean, this proof only requires that the empirical mean is close to the true mean for <span style="font-family: serif;">t = NK</span> and not all <span style="font-family: serif;">t &in; [T]</span>. But this extra condition will be helpful for further proofs.)</p>


Let the event that for all $$a\in[K]$$ and $$t\in[T]$$ , $$\lvert\mu_a - \overline \mu_{a,t}\rvert \leq \delta_{a,t}$$ be called clean event, and it occurs with probabilty $$\geq 1-O(\frac{1}{T^2})$$

The total regret suffered by the algorithm is the regret suffered in the exploration phase + the regret suffered in the exploitation phase. The regret in any 1 round of the exploration phase is bounded by the limits of rewards distribution *(6 - (-5))* and in the exploitation phase, with high probability *(clean event occurred)* is no more than $$\max_{a} 2\delta_{a,NK}$$ and with small probability *(clean even didn't happen)* is bounded by the limit of rewards distribution.

$$\implies \mathbb{E}[R] \leq N\cdot K\cdot 11 + (1-O(\frac{1}{T^2}))\cdot\max_{a} 2\delta_{a,NK} + O(\frac{1}{T^2})\cdot 11$$

 to minimize the above equation we can assume $$N$$ to be $$O\big((\frac{T^2\log T}{K^2})^\frac{1}{3}\big)$$ and $$\delta_{a,NK}$$ for any arm is $$O\big(\sqrt{ \frac{2\cdot121\cdot\log(T)}{N} }\big)$$

$$\implies \mathbb{E}[R] = O\big( T^{\frac{2}{3}}K^\frac{1}{3}(\log T)^\frac{1}{3} \big)$$

<p style="text-align: right;">&#9633;</p>


A Python code implementation of the above algorithm looks like:
```python
class Explore_then_commit:
	def __init__(self,actions, T):
		self.k = len(actions)
		self.N = (T**(2/3))*(np.log(T)**1/3)*self.k
	def get_weights(self,actions, history, reward_dict, T):
		t = len(history)
		if len(history) < self.N:
			return [1 if arm == actions[t%self.k] else 0 for arm in actions ]
		best_arm = max( actions , key = lambda action : reward_dict[action]['sum_rewards']/reward_dict[action]['num_pulls'] )
		weights = [1 if action == best_arm else 0 for action in actions]
		return np.array(weights)
	def next_action(self, actions , history , reward_dict , T):
		return choose_action(self.get_weights(actions,history,reward_dict,T))	
	def update(self, chosen_action, reward, history , reward_dict, T):
		pass
```
The implementation of the function choose action is as follows:
```python
def choose_action( weights ):
	weights = np.array(weights)/sum(weights)
	k = len(weights)
	choice = np.random.choice( list(range(k)) , p=weights)
	return choice
```
Here, 
+ np is the library numpy
+ the variable actions is the set $$[K]$$,
+ history is a list of tuple of $$(a_t,X_{a_t,t})$$
+ reward_dict is a dictionary storing the number of pulls and the sum of rewards for each action.
+ The function get_weights at any point returns the probabilities with which the algorithm chooses to select any action at a time instance. 
*Here the algorithm is deterministic so this function seems pointless but its utility will get clear in some time*
+ next_action returns the next action
+ The update function is to update any internal state after observing the reward for the chosen action.
&nbsp;
&nbsp;
&nbsp;

#### $$\epsilon$$-Greedy:


Another algorithm that achieves the same expected regret is the $$\epsilon$$-Greedy algorithm.
In each round, the algorithm chooses the arm with the highest empirical award with probability $$1-\epsilon_t$$, and with probability $$\epsilon_t$$ it chooses a random arm. With  $$\epsilon_t = O((\frac{K\log(t)}{t})^\frac{1}{3})$$ 

it achieves similar as explore_then _commit regret bound, 

$$\mathbb{E}[R] = O\big( T^{\frac{2}{3}}K^\frac{1}{3}(\log T)^\frac{1}{3} \big)$$

Proof Sketch: For any round $$t$$, we can expect that with a high probability that clean event occurs and that all arms would be pulled more than $$\frac{1}{2}\times\frac{\epsilon_t t}{k}$$.

Hence regret in round $$t$$ in expectation would be bounded by 

$$R_t \leq \Delta_{a_t} + \epsilon_t \cdot \max [\Delta_a] $$

Since $$a_t$$ has highest empirical reward, we can use the clean event condition and argue that 

$$\lvert\mu_{a*} - \mu_{a_t}\rvert \le \delta_{a_t} + \delta_{a*} \leq \sqrt{\frac{2\cdot121\log t}{n_t(a*)}}+\sqrt{\frac{2\cdot121\log t}{n_t(a_t)}}$$

$$\implies \mathbb{E}[R_t] \leq \epsilon_t\max_a \Delta_a + 2\sqrt{\frac{2\cdot2\cdot121\cdot k\log t}{\epsilon_t k}}$$

Substituting the value of $$\epsilon_t$$ that minimizes the expression on RHS and then summing this value for all $$t\in [T]$$ gives the expected regret bound.

<p style="text-align: right;">&#9633;</p>

A Python code implementation of the above algorithm looks like: 
``` python
class EGreedy:
	def __init__(self, actions, T):
		self.k = len(actions)
		self.epsilon = lambda t: (self.k * np.log(t) / t)**(1/3)
	def get_weights(self, actions, history, reward_dict, T):
		if len(history) < self.k:
			unchosen_actions = [el for el in actions if el not in [h[0] for h in history ] ]
			return [1 if i in unchosen_actions else 0 for i in range(self.k)] 		
		t = len(history)+1
		best_arm = max( actions , key = lambda action : reward_dict[action]['sum_rewards']/reward_dict[action]['num_pulls'] )
		weights = (1-self.epsilon(t))*np.array([1 if action == best_arm else 0 for action in actions])
		weights += self.epsilon(t)*np.ones(self.k)
		return weights
	def next_action(self, actions , history , reward_dict , T):
		return choose_action(self.get_weights(actions,history,reward_dict,T))	
	def update(self, chosen_action, reward, history , reward_dict, T):
		pass
```
&nbsp;
&nbsp;
&nbsp;

#### Successive_Elimination:


One drawback with the previous two algorithms is that the exploration phase is completely oblivious to the rewards observed. If certain arms have already shown that they have very bad rewards compared to others, there is no need to keep trying them out.
Based on this idea here is an algorithm that achieves better regret guarantees.

We have earlier shown that with a very high probability, the empirical mean and the true mean are within $$\delta_{a,t} = \sqrt{\frac{2\cdot121\cdot\log T}{n_t(a)}}$$ difference. Based on this, for any arm at any time instance, we assume a window where its true mean lies.  $$\mu_a \in [\overline\mu_{a,t} - \delta_{a,t} , \overline\mu_{a,t} + \delta_{a,t}]$$
If at any time, we observe that the upper limit of the true mean for an arm is less than the lower limit of the true mean of another arm, we can safely assume that this arm is not the optimal arm.
The algorithm protocol is as follows: initially, all arms are active. If the upper limit of the mean reward of an arm becomes less than the lower limit of the mean reward of any other arm, mark this arm as inactive. Select any active arm at random.
This algorithm achives $$O(\sqrt{KT\log T})$$ regret bound.
Proof Sketch: For any arm $$a_0$$, that was last pulled at time $$t_0$$, then 

$$\lvert\mu_{a*} - \mu_{a_0} \rvert \leq \delta_{a_0,t}+\delta{a* ,t} \leq \sqrt{\frac{2\cdot121\cdot\log T}{n_{t_0}(a_0)}}+\sqrt{\frac{2\cdot121\cdot\log T}{n_{t_0}(a*)}}$$

Now, $$n_{t_0}(a*) \approx n_{t_0}(a_0) = n_{T}(a_0)$$.

Hence, $$\lvert\mu_{a*} - \mu_{a_0} \rvert \leq O(\sqrt{\frac{\log T}{n_{T}(a_0)}})$$.
The total regret suffered from by pulling this arm multiple times is $$\leq O(\sqrt{\frac{\log T}{n_{T}(a_0)}})\times n_{T}(a_0)$$. On summing this quantity for all arms, the total regret for this algorithm can be bounded.

<p style="text-align: right;">&#9633;</p>

A Python code implementation of the above algorithm looks like:
```python
class Successive_elimination:
	def __init__(self,actions,T):
		self.k = len(actions)
		self.active_actions = actions.copy()
		self.radius = lambda n : np.sqrt(2*np.log(T)/n)
	def get_weights(self, actions, history, reward_dict, T):
		if len(history) < self.k:
			unchosen_actions = [el for el in actions if el not in [h[0] for h in history ] ]
			return [1 if i in unchosen_actions else 0 for i in range(self.k)]
		return [1 if i in self.active_actions else 0 for i in range(self.k)]
	def next_action(self, actions, history, reward_dict, T):
		return choose_action(self.get_weights(actions, history, reward_dict, T))
	def update(self, chosen_action, reward, history , reward_dict, T):
		if len(history) < self.k or len(self.active_actions) <= 1:
			return
		highest_lower_bound = max( [ reward_dict[action]['sum_rewards']/reward_dict[action]['num_pulls'] - self.radius(reward_dict[action]['num_pulls']) for action in self.active_actions ] )
		remaining_actions = [ action for action in self.active_actions if reward_dict[action]['sum_rewards']/reward_dict[action]['num_pulls'] + self.radius(reward_dict[action]['num_pulls']) > highest_lower_bound ]
		self.active_actions = remaining_actions
		return
```
&nbsp;
&nbsp;
&nbsp;


#### UCB1:
Let us consider another approach with adaptive exploration. 
Assume that every arm is as good as it can possibly be and choose the arm with the highest upper limit on mean reward in each round.
The two summands in the estimation of the choice arm, $$\overline\mu+ \delta$$ strike a balance between exploration and exploitation, the former increases for good arms and the latter for less explored arms.
This algorithm achieves an expected regret of $$O(\sqrt{KT\log T})$$

Proof Sketch:
The proof idea is the same as Successive elimination in a round $$t_0$$ if an arm $$a_0$$ is selected, then the regret with this round can be bounded with $$O(\sqrt{\frac{\log T}{n_{t_0}(a_0)} })$$ with high probability and then the by summing this bound over all rounds, the total regret can be bounded with $$O(\sqrt{KT\log T})$$.

A Python code implementation of the above algorithm looks like:
```python
class UCB1:
	def __init__(self, actions, T):
		self.k = len(actions)
		self.radius = lambda n : np.sqrt(2*np.log(T)/n)
	def get_weights(self, actions, history, reward_dict, T):
		if len(history) < self.k:
			unchosen_actions = [el for el in actions if el not in [h[0] for h in history ] ]
			return [1 if i in unchosen_actions else 0 for i in range(self.k)]
		chosen_action = max(reward_dict , key = lambda action: reward_dict[action]['sum_rewards']/reward_dict[action]['num_pulls']+self.radius(reward_dict[action]['num_pulls']))
		return [1 if i == chosen_action else 0 for i in actions]
	def next_action(self, actions, history, reward_dict, T):
		return choose_action(self.get_weights(actions,history,reward_dict,T))
	def update(self, chosen_action, reward, history , reward_dict, T):
		return
```
&nbsp;
&nbsp;
&nbsp;


### Adversarial Bandits:

Stochastic Bandits take a very strong assumption on the rewards of the actions. This strong assumption might limit the application and guarantees of the above algorithms where the i.i.d. assumption is violated. Adversarial bandits swing on the other side and take no assumption on the reward of each arm. 

| **Protocol:** |
|------------------|
| ***Parameters***: $$K$$ arms, $$T$$ rounds, $$T > K$$, for any time $$t\in[T]$$ the environment chooses reward $$X_{a,t}$$ for each arm $$a \in [K]$$, The environment can adapt the reward based on the algorithm protocol and the history of all chosen actions and rewards |
|For each round $$t \in [T]$$ the algorithm chooses an $$a_t\in[K]$$ and observes a reward $$X_{a_t,t}$$ |

No deterministic algorithm can hope to do well in this scenario, since the environment knows exactly which action the algorithm is going to choose apriori setting the rewards for the next round and set the rewards accordingly.
Even in this pessimistic scenario, we can guarantee some upper bounds on the regret suffered.
In adversarial bandits, pseudo-regret is defined as the deficit suffered from the best arm policy. 
$$E[R] = \max_{a\in[K]}\mathbb{E}\big[ \sum_{t\in[T]}X_{a,t} - \sum_{t\in[T]}X_{a_t,t} \big]$$ 
&nbsp;
&nbsp;
&nbsp;


#### EXP3:
Let's define the loss of any reward as the difference between the maximum possible reward and the observed reward, scaled between 0 to 1, for our scenario, our rewards are bounded between -5 and 6, so our 

$$loss = \frac{-1}{11}\cdot(reward-6)$$ 

The key idea in the exp3 algorithm is it tries to keep a record of cumulative loss for each arm. Since in any round reward/loss of only one action is observed keeping the exact record is not possible. The idea is to scale the loss with the inverse of the probability of observing it. 

The loss of action $$a$$ at round $$t$$ is 


$$\tilde l_{a_0,t} = \begin{cases}
    l_{a_t,t}/\mathbb{P}[a_t = a_0] & \text{if } a_t = a_0 \\ % & is your "\tab"-like command (it's a tab alignment character)
    0 & \text{otherwise.}
\end{cases}$$


Here, $$\tilde l_{a_0,t}$$ can be calculated for all arms $$a\in[K]$$ irrespective if it were pulled or not.
The cummulative loss is $$\tilde L_{a,t} = \sum_{i\leq t}\tilde l_{a,i}$$.

The expected value of $$\tilde l_{a,t}$$ in round $$t$$ is equal to the actual loss of the arm $$l_{i,t}$$.

(*Let $$P_{a,t}$$ be $$\mathbb{P}[a_t = a]$$.*)

$$\implies \mathbb{E}[ \tilde l_{a_0,t}] = \sum_{a\in[K]} P_{a,t} \times \frac{l_{a,t}\cdot \mathbb{I}\{a=a_0\}}{P_{a,t}} = l_{a_0,t}$$ and $$\mathbb E[\tilde l_{a,t}^2] = \frac{l_{a,t}^2}{p_{a,t}}$$

After the cumulative loss of each arm is known, the probability distribution for selecting an arm $$a_0$$ is an exponential weight of the loss of that arm.

$$P_{a_0,t} = \frac{\exp(-\eta_t \tilde L_{a_0,t}) }{\sum_a \exp(-\eta \tilde L_{a,t})}$$ for $$\eta \geq 0$$.

For large $$\eta$$ the algorithm tends to exploit the result and aggressively choose arms with less cumulative loss, on the other hand with smaller $$\eta$$ it tends to explore more, and when $$\eta = 0$$ it chooses all arms with equal likelihood.

If exp-3 is run with $$\eta = \sqrt{\frac{2\log K}{TK} }$$ the pseudo-regret is $$\mathbb E[R] = O(\sqrt{TK\log K})$$.

Proof Sketch:  Let $$w_{a,t} = e^{-\eta \tilde L_{a,t}}$$ and $$W_{t} = \sum w_{a,t} = \sum e^{-\eta \tilde L_{a,t}}$$

The proof is using the idea that we can lower bound the total reduction in  $$W_T$$ with the total loss incurred for any one arm. And for each round, whenever some loss $$l_{a_t,t}$$ is observed, $$W_t$$ can be upper bounded by the amount it reduces. Using the two bounds, we can show that the total loss incurred by the algorithm is not much larger than the loss incurred by the best arm.

Here is how the proof flows:

We can upper bound $$\frac{W_T}{W_0}$$ by the total Loss of any arm as: 

$$\log(\frac{W_T}{W_0}) = \log(\frac{\sum_a e^{-\eta \tilde L_{a,T}}}{K}) \geq -log K - \eta \tilde L_{a,T}\text{        }\forall a\in[K]$$

and for any one round $$t$$, we can lower bound $$\frac {W_t}{W_{t-1}}$$ by the loss incurred in that round by:

$$\log(\frac{W_t}{W_{t-1}}) = \log{( \sum_a \frac{w_{a,t}}{W_{t-1}}\cdot e^{-\eta\tilde l_{a,t}}  )}=\log{( \sum_a p_{a,t}\cdot e^{-\eta\tilde l_{a,t}}  )}$$ 

$$\leq \log{( \sum_a p_{a,t}\cdot (1-\eta\tilde l_{a,t}+\frac{\eta^2}{2}\tilde l_{a,t}^2)  )}\leq \sum_a p_{a,t}(-\eta\tilde l_{a,t}+\frac{\eta^2}{2}\tilde l_{a,t}^2)$$

This is using: $$e^x \leq 1+x+\frac{x^2}{2}\forall x \leq 0\text{ and }e^x \geq 1+x$$

&nbsp;
&nbsp;
By using the previous bound along with repeated application of the above bound for all $$t$$.

We get that  $$\forall a: -log K - \eta \tilde L_{a,T} \leq \sum_{t\in[T]}\sum_b p_{b,t}(-\eta\tilde l_{b,t}+\frac{\eta^2}{2}\tilde l_{b,t}^2)$$

rearranging the term in the equation, we get:

$$\sum_{t\in[T]}\sum_{b\in[K]}p_{b,t}\tilde l_{b,t} - \sum_t \tilde l_{a,t} \leq \frac{\log K}{\eta} + \frac{\eta}{2}\sum_{t\in[T]}\sum_{b\in[K]}p_{b,t}\tilde l_{b,t}^2\text{  } \forall a$$

Now, using the expected value, 

$$\mathbb E[\sum_{t\in[T]}\sum_{b\in[K]}p_{b,t}\tilde l_{b,t} - \sum_t \tilde l_{a,t}] = \mathbb E[\sum_t l_{a_t,t} - \sum_t l_{a,t}]$$ 

and definition of regret $$R = max_a\mathbb E[\sum l_{a_t,t} - \sum l_{a,t}]$$

we get the following bound on the regret. 

$$\implies R \leq \frac{\log K}{\eta} + \frac{\eta}{2}\sum_t\sum_b p_{b,t}\mathbb E [\tilde l_{b,t}^2] = \frac{\log K}{\eta} + \frac{\eta}{2}\sum_t\sum_b l_{b,t}^2 = \frac{\log K}{\eta} + \frac{\eta}{2}KT\Delta \text{ ; where }\Delta = \max_a \Delta_a$$


A Python code implementation of the above algorithm looks like:
```python
class EXP3:
	def __init__(self,actions,T):
		self.k = len(actions)
		self.L = np.zeros(self.k)
		self.p = None
		self.eta = np.sqrt( np.log(self.k) / (T*self.k)  )
	def get_weights(self, actions, history, reward_dict, T):
		t = len(history)+1
		return [ np.exp(-1 * self.L[i]*self.eta) for i in range(self.k)  ]
	def next_action(self, actions , history , reward_dict , T):
		self.p = np.array(self.get_weights(actions, history, reward_dict, T))
		self.p /= np.sum(self.p)
		return choose_action(self.p)
	def update(self, chosen_action, reward, history , reward_dict, T):
		loss = (-1/11)*(reward-6)
		self.L[chosen_action] += loss / self.p[chosen_action]
		self.p = None
		return
```

For a better understanding and complete proofs of the above algorithms, we refer to these three surveys:

**Bandits, Multi-Armed**. (n.d.). *Introduction to Multi-Armed Bandits*.

**Bubeck, Sébastien, & Cesa-Bianchi, Nicolo**. (2012). *Regret analysis of stochastic and nonstochastic multi-armed bandit problems*. *Foundations and Trends® in Machine Learning*, 5(1), 1--122. Now Publishers, Inc.

**Lattimore, Tor, & Szepesvári, Csaba**. (2020). *Bandit algorithms*. Cambridge University Press.

&nbsp;
&nbsp;
&nbsp;


### Stochastic Bandits with Adversarial Corruption:
Both Stochastic and adversarial bandits swing too far with their assumptions on the reward scenarios. Here we want to discuss the work of Lykouris, Mirrokni, and Paes Leme (2018)

{Lykouris, Thodoris, Mirrokni, Vahab, & Paes Leme, Renato. (2018). *Stochastic bandits robust to adversarial corruptions*. In *Proceedings of the 50th Annual ACM SIGACT Symposium on Theory of Computing* (pp. 114–122).}

Another approach to the reward assumption can be in a middle ground, in this model we assume that rewards are initially drawn from a fixed distribution, but are adulterated with a finite amount of corruption by an adaptive adversary. 

| **Protocol:** |
|------------------|
| ***Parameters***: $$K$$ arms, $$T$$ rounds, $$T > K$$, for any time $$t\in[T]$$ the environment chooses reward $$X_{a,t}$$ for each arm $$a \in [K]$$ from distribution $$D_a$$. The adversary chooses a corruption $$c_{a,t}$$ based on the algorithm protocol and the history of all chosen actions and rewards. The total corruption is bounded: $$\sum_t max_a \lvert c_{a,t}\rvert \text{  }\leq C$$.
|For each round $$t \in [T]$$ the algorithm chooses an $$a_t\in[K]$$ and observes the corrupted reward $$X_{a_t,t}+c_{a,t}$$ |

By varying the total corruption this model generalizes both stochastic and adversarial bandits.
&nbsp;
&nbsp;
&nbsp;


#### Multi-layer Active Arm Elimination Race:
This algorithm extends successive elimination. In successive elimination, we kept a record of all active arms and eliminated arms, under the assumption of clean event if they could not be the best arms. The idea is similar, but instead of keeping a single such list, the algorithm maintains multiple such lists, let's call them layers(assume $$n$$ layers). In each round, it selects a layer with probability $$\propto_{\approx} 2^{-n}$$. The layer n updates its dictionaries of the number of arm pulls and empirical means, only when the layer is selected. The result of each subsequent layer is more robust to the corruption since they are likely to only admit about $$2^{-n}$$ times the corruption, in their dictionaries. Whenever a layer $$n$$ concludes that an arm $$a$$ needs to be eliminated from all its previous layers, also remove that arm. In case a layer is selected but has no active arms, it selects the arm based on the next smallest layer that is not empty.

A Python code implementation of the above algorithm looks like:
```python
class MultiLayer_active_arm_elimination:
	def __init__(self,actions,T):
		self.k = len(actions)
		self.num_layers = int(np.ceil(np.log(T)))
		self.sum_rewards = np.zeros( (self.num_layers,self.k) )
		self.num_pulls = np.zeros((self.num_layers,self.k))
		self.active_arms = [ actions.copy() for i in range(self.num_layers)  ]
		self.chosen_layer = None
		alpha = lambda n : np.log(4* self.k * T * np.log(T) / 0.01 ) / n
		self.radius = lambda n : alpha(n)+ np.sqrt(alpha(n))
	def get_weight_nthlayer(self, n, actions, history, reward_dict, T):
		if len(self.active_arms[n]) == 0:
			return self.get_weight_nthlayer(n+1, actions, history, reward_dict, T)
		return np.array([1 if i in self.active_arms[n] else 0 for i in actions])	
	def get_weights(self, actions, history, reward_dict, T):
		weights = np.zeros(self.k)
		for i in range(self.num_layers):
			weights += 2**( -(i+1) ) * self.get_weight_nthlayer(i,actions,history,reward_dict,T)
		weights += 	2**( -(self.num_layers+1) )*self.get_weight_nthlayer(0,actions,history,reward_dict,T)
		return weights
	def next_action(self,actions,history,reward_dict,T):
		if len(history) < self.k:
			unchosen_actions = [el for el in actions if el not in [h[0] for h in history ] ]
			return choose_action([1 if i in unchosen_actions else 0 for i in range(self.k)])
		p = [ 2**(-1* (i+1) ) for i in range(self.num_layers) ]
		p[0] += 2**( -1*(self.num_layers) )
		self.chosen_layer = np.random.choice( range(self.num_layers)  , p = p )
		return choose_action( self.get_weight_nthlayer(self.chosen_layer, actions, history, reward_dict, T) )
	def update(self, chosen_action, reward, history , reward_dict, T):
		if len(history) < self.k:
			for layer in range(self.num_layers):
				self.num_pulls[layer][chosen_action]+=1
				self.sum_rewards[layer][chosen_action]+=reward
			self.chosen_layer = None
			return
		self.num_pulls[self.chosen_layer][chosen_action]+=1
		self.sum_rewards[self.chosen_layer][chosen_action]+=reward
		if len(self.active_arms[self.chosen_layer]) <= 1:
			self.chosen_layer = None
			return
		lower_bounds = [ (			
			self.sum_rewards[self.chosen_layer][arm]/self.num_pulls[self.chosen_layer][arm] 
			- self.radius(self.num_pulls[self.chosen_layer][arm])			
			)
			 for arm in self.active_arms[self.chosen_layer] ]
		highest_lower_bound = max(lower_bounds)
		self.active_arms[self.chosen_layer] = [arm for arm in self.active_arms[self.chosen_layer]
				if (
				self.sum_rewards[self.chosen_layer][arm]/self.num_pulls[self.chosen_layer][arm]
				 + self.radius(self.num_pulls[self.chosen_layer][arm] ) 
				 ) 
					> highest_lower_bound 
				]
		for i in range(self.chosen_layer):
			self.active_arms[i] = list(set(self.active_arms[i]) & set(self.active_arms[self.chosen_layer]))
		self.chosen_layer = None
		return
```
&nbsp;
&nbsp;
&nbsp;



### Simulations:

Now that we have introduced all the algorithms, let's see how they practically fare, by running some simulations.
&nbsp;
&nbsp;
&nbsp;


#### Stochastic Bandits:
Consider the case of when the rewards follow i.i.d. assumption at each time step.
The simulation code is as follows:
```python
def simulate_stochastic_bandits(k, T, mean_rewards , solver):
	assert k == len(mean_rewards)
	actions = list(range(k))
	reward_dict = {action: { 'num_pulls': 0, 'sum_rewards':0} for action in actions}
	# a list of tuple of action chosen and reward observed
	history = [  ]
	mab = solver(actions, T)
	for iteration in range(T):
		chosen_action = mab.next_action(actions,history,reward_dict,T)
	# sampling reward for the chosen action
		reward = np.random.normal(loc = mean_rewards[chosen_action] , scale = 1)
		reward = max(-5, reward)
		reward = min(6,reward)
		mab.update(chosen_action,reward,history,reward_dict,T)
	#updating the history of actions and rewards and dict of number of pulls and sum of rewards
		history.append( (chosen_action , reward) )
		reward_dict[chosen_action]['num_pulls']+=1
		reward_dict[chosen_action]['sum_rewards']+=reward
	regret = max(mean_rewards) - sum([el[1] for el in history])/T
	return( max(mean_rewards) , sum([el[1] for el in history])/T  , regret )
```

Here, history is a list of tuples of actions chosen and rewards observed, the reward dictionary maintains the number of pulls and sum of rewards for each action. The simulation returns the maximum expected reward for an arm, the average reward observed by the algorithm, and the average regret suffered by the algorithm.

Let's observe the regret for $$k = 3$$, with mean rewards be $$[0,0.3,1]$$ and $$T = 50000$$.
The results are:


|algorithm | Expected reward of optimal arm | Average reward in each round | average Regret in each round|
|---|---|---|---|
|Explore_then_commit|1.0|0.82844|0.17156|
|EGreedy|1.0|0.83584|0.16416|
|Successive_elimination|1.0|0.99825|0.00175|
|UCB1|1.0|0.99574|0.00426|
|EXP3|1.0|0.88704|0.11296|
|MultiLayer_active_arm_elimination|1.0|0.94698|0.05302|


&nbsp;
&nbsp;
&nbsp;
#### Stochastic Bandits with switched mean:
Let's simulate another scenario in which the mean rewards are switched after T/2 rounds.

The simulation code is as follows:
```python
def simulate_switchmean_stochastic_bandits(k,T,mean_rewards,switched_rewards,solver):
	assert k == len(mean_rewards) == len(switched_rewards)
	actions = list(range(k))
	reward_dict = {action: { 'num_pulls': 0, 'sum_rewards':0} for action in actions}
	history = []
	mab = solver(actions, T)
	for iteration in range(T):
		chosen_action = mab.next_action(actions,history,reward_dict,T)
		reward = np.random.normal(loc = mean_rewards[chosen_action] , scale = 1)
		if iteration > T//2:
			reward = np.random.normal(loc = switched_rewards[chosen_action] , scale = 1)
		reward = max(-5, reward)
		reward = min(6,reward)
		mab.update(chosen_action,reward,history,reward_dict,T)
		history.append( (chosen_action , reward) )
		reward_dict[chosen_action]['num_pulls']+=1
		reward_dict[chosen_action]['sum_rewards']+=reward
	regret =  max(mean_rewards) * (1/2) + max(switched_rewards)*(1/2) - sum([el[1] for el in history])/T
	return((max(mean_rewards) * (1/2) + max(switched_rewards)*(1/2) ),(sum([el[1] for el in history])/T),regret)
```
Let's observe the regret for $$k = 3$$, with mean rewards be $$[0,0.3,1]$$ and switched rewards be $$[1,0.3,0]$$ and $$T = 50000$$.
The results are:


|algorithm | Expected reward of optimal arm | Average reward in each round | average Regret in each round|
|---|---|---|---|
|Explore_then_commit|1.0|0.3263|0.6737|
|EGreedy|1.0|0.45929|0.54071|
|Successive_elimination|1.0|0.49763|0.50237|
|UCB1|1.0|0.99367|0.00633|
|EXP3|1.0|0.50991|0.49009|
|MultiLayer_active_arm_elimination|1.0|0.45982|0.54018|


&nbsp;
&nbsp;
&nbsp;
#### Adaptive Adversarial bandits:
Let's simulate the case where the adversary for the first T/10 rounds, simulates stochastic bandits, and then sets the reward for each arm either 0 or 1, depending if the probability of it getting pulled is greater or less than $$1/K$$.
The simulation code is as follows:
```python
def simluate_adaptive_adversarial_bandits(k,T,mean_rewards,solver):
	assert k == len(mean_rewards)
	actions = list(range(k))
	reward_dict = {action: { 'num_pulls': 0, 'sum_rewards':0} for action in actions}
	history = []
	rewards = []
	mab = solver(actions, T)
	for iteration in range(T):
		weights = np.array(mab.get_weights(actions,history,reward_dict,T), dtype=np.float64)
		weights /= sum(weights)
		if iteration < T/10:
			current_rewards = [ np.random.normal(loc = mean_rewards[i] , scale = 1) for i in range(k) ]
		else:
			current_rewards = [ 0 if i > 1/k else 1 for i in weights ]
		chosen_action = mab.next_action(actions,history,reward_dict,T)
		reward = current_rewards[chosen_action]
		reward = max(-5, reward)
		reward = min(6,reward)
		mab.update(chosen_action,reward,history,reward_dict,T)
		rewards.append(current_rewards)
		history.append( (chosen_action , reward) )
		reward_dict[chosen_action]['num_pulls']+=1
		reward_dict[chosen_action]['sum_rewards']+=reward
	rewards = np.array(rewards)
	sum_rewards = np.sum(rewards, axis = 0)
	regret = max(sum_rewards)/T - sum([el[1] for el in history])/ T
	return ((max(sum_rewards) / T),(sum([el[1] for el in history])/ T),regret)
```
Let $$k = 3$$, mean rewards be $$[0,0.3,1]$$ and T = 50000.
The results are:

|algorithm | Expected reward of optimal arm | Average reward in each round | average Regret in each round|
|---|---|---|---|
|Explore_then_commit|0.83454|0.04437|0.79016|
|EGreedy|0.68796|0.23788|0.45008|
|Successive_elimination|0.92911|0.09626|0.83285|
|UCB1|0.89915|0.09995|0.7992|
|EXP3|0.53629|0.51533|0.02097|
|MultiLayer_active_arm_elimination|0.65429|0.12936|0.52493|



&nbsp;
&nbsp;
&nbsp;
#### Stochastic bandits with finite corruption:
Let's simulate the case, where the rewards are drawn from an i.i.d. assumption, but an adversary can inject a finite amount of adversarial noise in order to increase the regret of the algorithm.
There can be many strategies for the adversary, but we are using a strategy that whenever the algorithm has greater than $$1/K$$ probability of choosing the best arm, it reduces its reward by 5 and increases the reward of all other arms by 5.
The simulation code is as follows:
```python
def simulate_stochastic_bandits_with_finite_corruption(k,T,mean_rewards,corruption_limit,solver):
	actions = list(range(k))
	reward_dict = {action: { 'num_pulls': 0, 'sum_rewards':0} for action in actions}
	history = []
	rewards = []
	corruption_used = 0
	mab = solver(actions, T)
	best_arm = mean_rewards.index(max(mean_rewards))
	for iteration in range(T):
		weights = np.array(mab.get_weights(actions,history,reward_dict,T), dtype=np.float64)
		weights /= sum(weights)
		current_rewards = [ np.random.normal(loc = mean_rewards[i] , scale = 1) for i in range(k) ]
		if weights[best_arm] >= 1/k and corruption_used < corruption_limit:
			# for i in range(1,k):
				# current_rewards[i] -= 1
			for i in range(0,k):
				if i != best_arm:
					current_rewards[i] += 5
			current_rewards[best_arm] -= 5
			corruption_used +=5
		chosen_action = mab.next_action(actions,history,reward_dict,T)
		reward = current_rewards[chosen_action]
		reward = max(-5, reward)
		reward = min(6,reward)
		mab.update(chosen_action,reward,history,reward_dict,T)
		rewards.append(current_rewards)
		history.append( (chosen_action , reward) )
		reward_dict[chosen_action]['num_pulls']+=1
		reward_dict[chosen_action]['sum_rewards']+=reward
	rewards = np.array(rewards)
	sum_rewards = np.sum(rewards, axis = 0)
	regret = max(sum_rewards) / T - sum([el[1] for el in history])/ T
	return ((max(sum_rewards) / T),(sum([el[1] for el in history])/ T),regret)
```
Lets, run with mean rewards as $$[0,0.3,1]$$, $$T = 50000$$ and corruption limit as $$1000$$
The results are:

|algorithm | Expected reward of optimal arm | Average reward in each round | average Regret in each round|
|---|---|---|---|
|Explore_then_commit|0.98127|0.81756|0.16371|
|EGreedy|0.98259|0.79073|0.19187|
|Successive_elimination|1.00005|0.29011|0.70994|
|UCB1|1.00494|0.2935|0.71144|
|EXP3|0.98156|0.90013|0.08142|
|MultiLayer_active_arm_elimination|0.97867|0.90995|0.06872|



&nbsp;
&nbsp;
&nbsp;
### Expected Regret vs Psuedo Regret:
Throughout this article, we used the algorithm objective to minimize pseudo-regret. Another similar objective could be to minimize expected regret.

Expected regret $$\mathbb E [R] = \mathbb E \big [\max_a\sum_tX_{a,t} - \sum_t X_{a_t,t}\big]$$.

Whereas our psedo regret is $$\overline R = \max_a \mathbb E\big[ \sum_t X_{a,t}  - \sum_{t}X_{a_t,t}\big]$$.

Since pseudo regret is the expected deficit from optimal action whereas expected regret is the expectation of regret with the action that is optimal.
The expected regret is a stronger notion and $$\overline R \leq \mathbb E [R]$$.
Here is a simulation to highlight that Expected regret is $$\geq$$ pseudo-regret.
it shows that for normal rewards with mean $$= [1 , 0.5, 0.99 , 0.9 , 0.2  , 0.1 , 0]$$

$$E[\max_a \sum_t X_{a,t}] \geq \max_a\mathbb E[\sum_t X_{a,t}]$$

```python
def expected_vs_psuedo_regret():
	T = 1000
	mean_rewards = [1 , 0.5, 0.99 , 0.9 , 0.2  , 0.1 , 0]
	k = len(mean_rewards)
	alpha = np.random.normal(loc = mean_rewards, scale = 1, size = (T,k))
	alpha = np.sum(alpha, axis = 0)
	alpha /= T
	expected_max_avg_value = max(alpha)
	max_expected_value = max(mean_rewards)
	return expected_max_avg_value - max_expected_value
```
Running this simulation 10000 times, average difference $$E[\max_a \sum_t X_{a,t}] - \max_a\mathbb E[\sum_t X_{a,t}]$$ was: 0.013302.

This happens because with arms that are close to optimal, like with mean 0.99, instead of 1, there is a chance that its sum of rewards is greater than the sum of rewards by the actual optimal arm. 

This difference vanishes as T increases, running the same simulation with $$T = 10000$$, 10 times the previous, the average difference was: 0.0019969. about a tenth of the previous difference.
