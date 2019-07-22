# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agent here was written by Simon Parsons, based on the code in
# pacmanAgents.py
# learningAgents.py

# This solution to the coursework, assigned by Simon Parsons, was written by Nghi Bao Le.

from pacman import Directions
from game import Agent
import random
import game
import util

# QLearnAgent
#
class QLearnAgent(Agent):

    # Constructor, called when we start running the
    def __init__(self, alpha=0.2, epsilon=0.05, gamma=0.8, numTraining = 10):
        # alpha       - learning rate
        # epsilon     - exploration rate
        # gamma       - discount factor
        # numTraining - number of training episodes
        #
        # These values are either passed from the command line or are
        # set to the default values above. We need to create and set
        # variables for them
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)
        # Count the number of games we have played:
        self.episodesSoFar = 0
        
        # Initialize Q-Dictionary to keep track of the State-Action pairs and values:
        
        self.q_dict = {} 
        
        # This is to keep track of previous state's score in order to calculate reward:
        
        self.prev_score = 0
        
        # This is to keep track of the time steps over an episode:
        
        self.num_iterations = 0
        
    
    # Accessor functions for the variable episodesSoFars controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar +=1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
            return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value):
        self.epsilon = value

    def getAlpha(self):
        return self.alpha

    def setAlpha(self, value):
        self.alpha = value
        
    def getGamma(self):
        return self.gamma

    def getMaxAttempts(self):
        return self.maxAttempts
    
    # greedyPick function to pick actions according to e-greedy mode.
    
    def greedyPick(self, legal_actions, state):
        
        # This function accepts a list, such as the list of legal actions, and 
        # return an action from it according to epsilon greedy pick.
        
        # Generate a random number as probability to implement epsilon greedy:
        
        random_number = random.uniform(0,1)
        
        # Filter the dictionary by the states, in order to pick out the actions.
        
        filter_dict = {k: v for k, v in self.q_dict.iteritems() if 
                       self.current_state in k}
        
        # Check if the random number is less than epsilon:
        if random_number <= self.epsilon:
            # Pick a random action from list of legal actions:
            action = random.choice(legal_actions)
        else:
            # Get the actions with the highest Q-value.
            # The random.choice is just in case there are more than one actions
            # with the same highest Q-values. Then we pick a random action from
            # those.
            action = random.choice([k for k,v in filter_dict.iteritems() 
                                    if v == max(filter_dict.values())])[-1]

        return action


    
    # getAction
    #
    # The main method required by the game. Called every time that
    # Pacman is expected to move
    def getAction(self, state):

        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
            
            
        #### SARSA implementation ####
        
        # 1. Create an empty dictionary to sequentially add in and keep track of
        # state-action pairs. For newly encountered State-Action pairs, we assign
        # them to 0.
        
        # 2. The key of the dictionary will be a tuple of (states, action) with
        # associated Q-Values
        
        # 3. We pick action using e-greedy. E-greedy means we pick the action 
        # that maximizes the q(s,a) with with 1 - epsilon probability, then with 
        # epsilon prob, a random action out of all possible action is chosen.
        
        # 4. Update the state-action value according to SARSA update rule.
        
        # 5. Repeat until S is terminal.
        
        
        # Get the Food position:
        
        # The if-else statement makes getting the food location only once per
        # training and testing.
        if self.num_iterations == 0 and self.episodesSoFar == 0:
            self.food = []
            for row_index, row in enumerate(state.getFood()):
                for col_index, col in enumerate(row):
                    if col == True:
                        self.food.append((row_index,col_index))
        else:
            pass
        
        
        # Make a tuple of the current states:
        
        self.current_state = (state.getPacmanPosition(), tuple(state.getGhostPositions()), 
                              tuple(self.food))
        
        # Add in newly encountered State Action pairs and assign them Q-Values of 0:
        
        for action in legal:
            if (self.current_state, action) not in self.q_dict.keys():
                self.q_dict.update({(self.current_state, action): 0})
        
        # Pick an action using epsilon-greedy
        
        self.pick = self.greedyPick(legal, state)
        
        # Get the reward. If this is the initial state, then reward is 0.
        
        self.reward = state.getScore() - self.prev_score
                
        # SARSA udpate:
        
        # Check if the current state is the initial state, if it is, we skip the update:
        if self.num_iterations == 0:
            pass
        else:
            Q_SA = self.q_dict[(self.prev_state, self.prev_Action)]
            Q_SA_prime = self.q_dict[(self.current_state, self.pick)]
            
            # The actual update:
            
            self.q_dict[(self.prev_state, self.prev_Action)] = (Q_SA 
                        + self.getAlpha() * (self.reward + self.getGamma() * Q_SA_prime - Q_SA))

        
        # Remember the previous iteration's states, actions and scores:
        
        self.prev_score = state.getScore()
        self.prev_state = self.current_state
        self.prev_Action = self.pick
        
        print "Legal moves: ", legal
        print "Pacman position: ", state.getPacmanPosition()
        print "Ghost positions:" , state.getGhostPositions()[0]
        print "Food locations: "
        print state.getFood()
        print "Score: ", state.getScore()
        
        # Keep track of the time-steps:
        
        self.num_iterations = self.num_iterations + 1
        
        # We have to return an action:
        
        return self.pick
            

    # Handle the end of episodes
    #
    # This is called by the game after a win or a loss.
    def final(self, state):
        
        print "A game just ended!"
        
        # Update the state before the terminal state:
        # Terminal state Q-Value will be defined as 0. So it does not have a role
        # in this updating step.
        
        self.reward = state.getScore() - self.prev_score
        
        Q_SA = self.q_dict[(self.prev_state, self.prev_Action)]
        
        self.q_dict[(self.prev_state, self.prev_Action)] = (Q_SA + 
                    self.getAlpha() * (self.reward - Q_SA))
        
        # Reset the numbers:
        
        self.reward = 0
        self.prev_score = 0
        self.num_iterations = 0
        
        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg,'-' * len(msg))
            self.setAlpha(0)
            self.setEpsilon(0)


