---
title: "MultiArmedBandits with Stochastic, Adversarial, and Adversarially Corrupted Stochastic Rewards"
permalink: "/MultiArmedBandits/"
layout: post
mathjax: true
---

The multi-armed bandit (MAB) is a classic problem in probability theory and statistics that models exploitation-exploration trade-off, leveraging choices that have proven effective in the past versus choosing new options that might provide better-unexplored trade-offs. Imagine a row of slot machines, each with a different and unknown distribution of paying out a reward. The goal is to select the arm of a machine at each time instance such that it maximizes the total reward over time. This article will discuss some known algorithms in MAB in stochastically chosen, adversarially chosen, and stochastically chosen but strategically corrupted reward scenarios and run simulations for the algorithms discussed.

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
#### Stochastic Bandits:

Stochastic bandits are a type of problem in which the reward of different actions are unrelated to each other, and the rewards of the same action at different time steps are an i.i.d distribution. 
| **Protocol:** |
|------------------|
| ***Parameters***: $$K$$ arms, $$T$$ rounds, $$T > K$$, for each arm $$a \in [K]$$, the reward for arm $$a$$ is drawn from distribution $$D_a$$. |
|For each round $$n \in [T]$$ the algorithm chooses an $$a_t\in[K]$$ and observes a reward $$X_{a_t,t}$$ |

The aim of the algorithm is to minimize the deficit suffered from not always choosing the arm, with the highest total expected reward.
Let's define this deficit as Pseudo-Regret: 
$$$
R = \max_{a\in [k]} \mathbb{E}\Big[\sum_{i\in[T]}X_{a,t} - \sum_{i\in[T]}X_{a_t,t} \Big]
$$$

For the purpose of this article, let's assume that all stochastic rewards are drawn from Normal Distribution with mean between 0 and 1, and capped between -5 and 6.

<sub><sup>Note: the probabilty that $$N(0,1) \in [-5,5] is > 10^6$$, so we can assume properties for both bounded and normal distribution as when required.</sub></sup>

##### Explore-then-commit:

Let's start with a simple algorithm: explore arms uniformly selecting each action $$N$$ times *(exploration phase)* the commiting to the best arm for the remaining $$T-NK$$ rounds *(exploitation phase)*.
This simple looking algorithm suffers from sub-linear i.e. o(T) regret.
Proof sketch: 
From [Hoeffding's inequality](https://en.wikipedia.org/wiki/Hoeffding%27s_inequality)[^1], we can infer that if $$N$$ is chosen large enough, then the mean reward for each arm estimated by sampling in the exploration phase is almost equal to the true mean reward of the arm. Then the total regret for exploration rounds is bounded by $$NK$$ times $$\max_{a\in[K]}\Delta_a$$ , and the total regret in exploration rounds should be close to negligiable.
[^1]: *[Hoeffding's inequality](https://en.wikipedia.org/wiki/Hoeffding%27s_inequality) is a type of [Concentration inequality](https://en.wikipedia.org/wiki/Concentration_inequality), such inequalities come in useful for proving bounds on various bandits algorithms.*

Let $$\mu_a$$ be true mean of arm $$a$$ and $$\overline{\mu}_{a,t}$$ be the mean estimated by sampling untill round $$t$$. 
Then, by [Hoeffding's inequality](https://en.wikipedia.org/wiki/Hoeffding%27s_inequality)
$$\implies\mathbb{P}\big[ | \mu_a - \overline{\mu}_{a,t} | > \delta_{a,t} \big] \leq  2e^{\frac{-2n_a(t)\cdot\delta_{a,t}^2}{(6 - (-5 ))^2}} = 2e^{\frac{-2n_a(t)\cdot\delta_{a,t}^2}{121}}$$
Let $$\delta_{a,t} = \sqrt{\frac{2\cdot 121\cdot log(T)}{n_a(t)}} \implies \mathbb{P}\big[ | \mu_a - \overline{\mu}_{a,t} | > \delta_{a,t} \big] \leq\frac{2}{T^4}$$

By Union Bound, $$\mathbb{P}\big[ \cup_{\forall a \in [K],t\in [T]} |\mu_a - \overline{\mu}_{a,t} | > \delta_{a,t}\big] \leq \sum_{t\in[T]}\sum_{a\in[K]} \frac{2}{T^4} \lt \frac{2}{T^4}$$[^2]
[^2]: For this particular proof we dont't need such a strict condition on difference between empirical and true mean, this proof only requires that the emipirical mean is close to true mean for $$t = NK$$ and not all $$t\in[T]$$. But this extra condition will be helpful for further proofs.


Let the event that for all $$a\in[K]$$ and $$t\in[T]$$ , $$|\mu_a - \overline \mu_{a,t}| \leq \delta_{a,t}$$ be called clean event, and it occurs with probabilty $$\geq 1-O(\frac{1}{T^2})$$

The the total regret suffered by the algorithm is the regret suffered in exploration phase + regret suffered in exploitation phase. The regret in any 1 round of exploration phase is bounded by the limits of rewards distribution *(6 - (-5))* and in exploitation phase, with high probability *(clean event occured)* is no more than $$\max_{a} 2\delta_{a,NK}$$ and with small probability *(clean even didn't happen)* is bounded by the limit of rewards distribution.
$$\implies \mathbb{E}[R] \leq N\cdot K\cdot 11 + (1-O(\frac{1}{T^2}))\cdot\max_{a} 2\delta_{a,NK} + O(\frac{1}{T^2})\cdot 11$$
 to minimize the above equation we can assume $$N$$ to be $$O\big((\frac{T^2\log T}{K^2})^\frac{1}{3}\big)$$ and  $$\delta_{a,NK}/$$ for any arm is $$O\big(\sqrt{ \frac{2\cdot121\cdot\log(T)}{N} }\big)$$
$$\implies \mathbb{E}[R] = O\big( T^{\frac{2}{3}}K^\frac{1}{3}(\log T)^\frac{1}{3} \big)$$
$$\square$$

A python code implementation of the above algorithm looks like:
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
+ reward_dict is a dictionary storing the number of pull and sum of rewards for each action.
+ The function get_weights at any point returns the probabilities with which the algorithm chooses to select any action at a time instance. 
*Here the algorithm is deterministic so this function seems pointless but its utility will get clear in some time*
+ next_action returns the next action
+ update function is to update any internal state after observing the reward for the chosen action.

##### $$\epsilon$$-Greedy:
Another algorithm that achieves same expected regret is the $$\epsilon$$-Greedy algorithm.
In each round, the algorithm chooses the arm with highest empirical award with probability $$1-\epsilon_t$$ and with probabilty $$\epsilon_t$$ it chooses a random arm. With  $$\epsilon_t = O((\frac{K\log(t)}{t})^\frac{1}{3})$$ 
it achieves similar as explore_then _commit regret bound, 
$$\mathbb{E}[R] = O\big( T^{\frac{2}{3}}K^\frac{1}{3}(\log T)^\frac{1}{3} \big)$$

Proof Sketch: For any round $$t$$, we can expect that with a high probability that clean event occurs and that all arms would be pulled more than $$\frac{1}{2}\times\frac{\epsilon_t t}{k}$$.
Hence regret in round $$t$$ in expectation would be bounded by 
$$R_t \leq \epsilon_t \max \Delta_a + \Delta_{a_t}$$
Since, $$a_t$$ has highest empirical reward, we can use the clean event condition and argue that $$|\mu_{a*} - \mu_{a_t}| \le \delta_{a_t} + \delta_{a*} \leq \sqrt{\frac{2\cdot121\log t}{n_t(a*)}}+\sqrt{\frac{2\cdot121\log t}{n_t(a_t)}}$$
$$\implies \mathbb{E}[R_t] \leq \epsilon_t\max_a \Delta_a + 2\sqrt{\frac{2\cdot2\cdot121\cdot k\log t}{\epsilon_t k}}$$
Substituting the value of $$\epsilon_t$$ that minimizes the expression on RHS and then summing this value for all $$t\in [T]$$ gives the expected regret bound.
$$\square$$
A python code implementation of the above algorithm looks like: 
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

##### Successive_Elimination:
One draw-back with the previous two algorithms is that the exploration phase is completely oblivious to the rewards observed. If certain arms have already shown that they have very bad rewards compared to others, there is no need to keep trying them out.
Based on this idea here is an algorithm that achieves better regret guarantees.

We have earlier shown that with a very high probability, the empirical mean and the true mean are within $$\delta_{a,t} = \sqrt{\frac{2\cdot121\cdot\log T}{n_t(a)}}$$ difference. Based on this, for any arm at any time instance, we assume a window where its true mean lies.  $$\mu_a \in [\overline\mu_{a,t} - \delta_{a,t} , \overline\mu_{a,t} + \delta_{a,t}]$$
If at any time, we observe that the upper limit of the true mean for an arm is less than the lower limit of true mean of another arm, we can safely assume that, this arm is not the optimal arm.
The algorithm protocol is as follows: initially all arms are active. If the upper limit of mean reward of an arm becomes less than the lower limit of mean reward of any other arm, mark this arm as inactive. Select any active arm at random.
This algorithm achives $$O(\sqrt{KT\log T})$$ regret bound.
Proof Sketch: For any arm $$a_0$$, that was last pulled at time $$t_0$$, then 
$$|\mu_{a*} - \mu_{a_0} | \leq \delta_{a_0,t}+\delta{a* ,t} \leq \sqrt{\frac{2\cdot121\cdot\log T}{n_{t_0}(a_0)}}+\sqrt{\frac{2\cdot121\cdot\log T}{n_{t_0}(a*)}}$$
Now, $$n_{t_0}(a*) \approx n_{t_0}(a_0) = n_{T}(a_0)$$.
Hence, $$|\mu_{a*} - \mu_{a_0} | \leq O(\sqrt{\frac{\log T}{n_{T}(a_0)}})$$.
The total regret suffered from by pulling this arm multiple times is $$\leq O(\sqrt{\frac{\log T}{n_{T}(a_0)}})\times n_{T}(a_0)$$. On summing this quantity for all arms, the total regret for this algorithm can be bounded.
$$\square$$
A python code implementation of the above algorithm looks like:
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
		if len(history) < self.k:
			return
		if len(self.active_actions) <= 1:
			return
		highest_lower_bound = max( [ reward_dict[action]['sum_rewards']/reward_dict[action]['num_pulls'] - self.radius(reward_dict[action]['num_pulls']) for action in self.active_actions ] )
		remaining_actions = [ action for action in self.active_actions if reward_dict[action]['sum_rewards']/reward_dict[action]['num_pulls'] + self.radius(reward_dict[action]['num_pulls']) > highest_lower_bound ]
		self.active_actions = remaining_actions
		return
```

##### UCB1:
Let us consider another approach for adaptive exploration. 
Assume that every arm is as good as it can possibly be and choose the arm with the highest upper limit on mean reward in each round.
The two summands in the estimation of choice arm, $$\overline\mu+ \delta$$ strikes balance between exploration and exploitation, the former increases for good arms and the latter for less explored arms.
This algorithm achieves an expected regret of $$O(\sqrt{KT\log T})$$
Proof Sketch:
The proof idea is same as Successive elimination in a round $$t_0$$ if an arm $$a_0$$, then the regret with this round can be bounded with $$O(\sqrt{\frac{\log T}{n_{t_0}(a_0)} })$$ with high probability and then the by summing this bound over all rounds, the total regret can be bounded with $$O(\sqrt{KT\log T})$$.

A python code implementation of the above algorithm looks like:
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
#### Adversarial Bandits:

Stochastic Bandits take a very strong assumption on the rewards of the actions. This strong assumption might limit the application and guarantees of the above algorithms where the i.i.d. assumption is violated. Adversarial bandits swing on the other side and takes no assumption on the reward of each arms. The rewards could be chosen in advance for all rounds and all actions by an adversary, or an adversary could choose future rewards based on action selection and rewars in the past. Even in this pessimistic scenario we can guarantee some upper bounds on the regret suffered.
In adversarial bandits, pseudo-regret is defined as the deficit suffered from the best arm policy. $$E[R] = \max_{a\in[K]}\mathbb{E}\big[ \sum_{t\in[T]}X_{a,t} - \sum_{t\in[T]}X_{a_t,t} \big]$$ 
##### EXP3:
Let's define the loss of any reward as the difference between maximum possible reward and the observed reward, scaled between 0 to 1, for our scenario, our rewards are bounded between -5 and 6, so our $$loss = \frac{-1}{11}\cdot(reward-6)$$ 
The key idea in exp3 algorithm is it tries to keep a record of cumulative loss for each arm. Since in any round reward/loss of only one action is observed keeping the exact record is not possible. The idea is to scale the loss with inverse of the probability of observing it. 
The loss of action $$a$$ at round $$t$$ is 
$$$
\tilde l_{a_0,t} = \begin{cases}
    l_{a_t,t}/\mathbb{P}[a_t = a_0] & \text{if } a_t = a_0 \\ % & is your "\tab"-like command (it's a tab alignment character)
    0 & \text{otherwise.}
\end{cases}
$$$
Here, $$\tilde l_{a_0,t}$$ can be calculated for all arms $$a\in[K]$$ irrespective if it were pulled or not. The cummulative loss is $$\tilde L_{a,t} = \sum_{i\leq t}\tilde l_{a,i}$$.
The expected value of $$\tilde l_{a,t}$$ in round $$t$$ is equal to the actual loss of the arm $$l_{i,t}$$.
(*Using $$P_{a,t}$$ for $$\mathbb{P}[a_t = a]$$.*)
$$\implies \mathbb{E}[ \tilde l_{a_0,t}] = \sum_{a\in[K]} P_{a,t} \times \frac{l_{a,t}\cdot \mathbb{I}\{a=a_0\}}{P_{a,t}} = l_{a_0,t}$$
After the cummulative loss of each arm is known, the probability distribution for selecting an arm $$a_0$$ of 





A python code implementation of the above algorithm looks like:
```python
class EXP3:
	def __init__(self,actions,T):
		self.k = len(actions)
		self.L = np.zeros(self.k)
		self.p = None
	def get_weights(self, actions, history, reward_dict, T):
		t = len(history)+1
		eta = np.sqrt( np.log(self.k) / (t*self.k)  )
		return [ np.exp(-1 * self.L[i]*eta) for i in range(self.k)  ]
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








&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;

The Monty Hall Problem is one of the most nefarious problems in stochastic mathematical modelling. It is a simple looking problem that perplexes most people when confronted with its dilemma. This article will try to generalize the MHP on how the host of the show chooses his door to better understand the problem.


Let's consider, you are on a game show and the following happens:

1. You are presented with 3 doors(Let {$$A$$, $$B$$, $$C$$} ) in front of you and told that exactly one of them hides a prize and other two are empty your task is to select the door with the prize.
2. After you have made your selection, the host of the show selects a door that is not the same doors as yours.
3. The host's door is opened and it is found that, that door does not hide the prize.
4. You are offered to reconsider your option to the other unopened door or you can stick with your current door, before your door is opened and it is revelaed whether your task was succesful.


What should you do? which door has higher chances of containing  the prize.


The above problem is a slightly perturbed version of the MHP, in the monty hall problem, you are given with a crucial piece of inforamtion about step 2 i.e. how the host chooses his door.


Consider a not-Monty Hall problem , where the host is also unaware of the location of the prize and randomly chooses one of the other doors you haven't chosen. When you observe that this door does not contain the prize, how does your random variable update with this new information. (Lets say you had chosen the door A)


This new probability distribution can be easily calculated by the the concept of Conditional Random Variable.


$$=P( A \text{ is prize | One of randomly chosen B/C door is empty} )$$

$$= \frac{P( A \text{ is prize} \cap\text{ One of randomly chosen B/C door is empty}  )}{P(\text{One of randomly chosen B/C door is empty})}$$ 

$$= \frac{P( A\text{ is prize})}{P(\text{One of randomly chosen B/C door is empty})}$$

$$= \frac{1/3}{2/3} = 1/2$$

And thus, in this case it does not matter if you choose to switch or not since either way your winning odds are at half.


In the actual monty hall problem, the host is aware of the prize and delibrately chooses a door which does not contain the prize. In this case, 


$$= P( A \text{ is prize | there exists a door not A, that does not have the prize} )$$

$$= \frac{P( A\text{ is prize }\cap\text{ there exists a door not A, that does not have the prize } )}{P(\text{there exists a door not }A\text{, that does not have the prize})}$$

$$= \frac{P( A\text{ is prize})}{P(\text{there exists a door not A, that does not have the prize})}$$ 

$$= \frac{1/3}{1} = 1/3$$


In this case, the probability of your previously selected door containing the gift is $$1/3$$, and hence, you chances of winning are higher if you choose to switch.





Lets verify this argument via a simulation, in case of monty hall and not-monty hall problem.


For the not-monty hall problem:
{% highlight python %}

def not_monty_hall_sim():
	doors = ['a','b','c']
	#Let the prize be randomly chosen
	prize = random.choice( doors )
	#Your randomly chosen initial guess
	contestant_choice = random.choice(doors)
	#Host randomly choosing one of the remaining doors.
	host_choice = random.choice(list(filter(lambda x:x!=contestant_choice, doors)))
	#This is to satify the conditional information given to us.
	# since we know that when the host choice was revealed it was not the prize doors
	if (  host_choice == prize ):
		# Abort the simulation
		return (0,0)
	#Returing the output, the first index is if the inital guess was correct, and the second index is if this is a valid iteration. 
	if 	contestant_choice != prize:
		return (1 , 1)
	else:
		return (0 , 1)	
{% endhighlight %}



For the standard monty hall problem

{% highlight python %}
def monty_hall_sim():
	doors = ['a','b','c']
	#Let the prize be randomly chosen
	prize = random.choice( doors )
	#Your randomly chosen initial guess
	contestant_choice = random.choice(doors)
	#Host randomly choosing one of the remaining doors, which is not the prize door
 	#Since, host is aware of the prize door and is deliberately choosing one that is not the prize door.
	host_choice = random.choice(list(filter(lambda x:x!=contestant_choice and x!= prize, doors)))
	#This is to satify the conditional statement given to us.
 	# since we know that when the host choice was revealed it was not the prize doors
	if (  host_choice == prize ):
		# Abort the simulation
		return [0,0]
	#Returing the output, the first index is if the inital guess was correct, and the second index is if this is a valid iteration. 
	if 	contestant_choice != prize:
		return [1 , 1]
	else:
		return [0 , 1]	
{%endhighlight%}

Run the above simulation a large number of times to convince of correct probabilities.

{% highlight python %}
ans = [0,0]
for i in range(10000):
	t = (not_)monty_hall_sim()
	ans[0] += t[0]
	ans[1] += t[1]
print("Winning Prob if not switching: " , ans[0]/ans[1])
{%endhighlight%}
