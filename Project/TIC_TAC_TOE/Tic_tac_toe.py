# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 23:56:52 2017

"""

'''
I followed some steps from the link given below to guide me for the Tic Tac Toe Environment. 
But I didn't copy the code as it is from there.
The link consists a code for tic tac toe game using Artificial Intelligence Algorithm for playing against the Human.
My code uses Reinforcement learning algorithm for playing the game.

Link: https://inventwithpython.com/chapter10.html
'''
import numpy as np
import matplotlib.pyplot as plt

LENGTH = 3


# this class represents a tic-tac-toe game
class Environment:
  def __init__(self):
    self.board = np.zeros((LENGTH, LENGTH))
    self.x = -1 # represents an x on the board, player 1
    self.o = 1 # represents an o on the board, player 2
    self.winner = None
    self.ended = False
    self.num_states = 3**(LENGTH*LENGTH)

  def is_empty(self, i, j):
    return self.board[i,j] == 0

  def reward(self, sym):
    # no reward until game is over
    if not self.game_over():
      return 0

    # if we get here, game is over
    # sym will be self.x or self.o
    if self.winner == sym:
        return 1 
    #reward of 0.75 if draw. self.winner=2 if game draws
    elif self.winner == 2:
        return 0.7
    else:
        return 0

  def get_state(self):
    # returns the current state, represented as an int

    k = 0
    h = 0
    for i in range(LENGTH):
      for j in range(LENGTH):
        if self.board[i,j] == 0:
          v = 0
        elif self.board[i,j] == self.x:
          v = 1
        elif self.board[i,j] == self.o:
          v = 2
        h += (3**k) * v
        k += 1
    return h

  def game_over(self, force_recalculate=False):
    '''
    returns true if game over (a player has won or it's a draw)
    otherwise returns false
    also sets 'winner' instance variable and 'ended' instance variable
    '''
    if not force_recalculate and self.ended:
      return self.ended
    
    # check rows
    for i in range(LENGTH):
      for player in (self.x, self.o):
        if self.board[i].sum() == player*LENGTH:
          self.winner = player
          self.ended = True
          return True

    # check columns
    for j in range(LENGTH):
      for player in (self.x, self.o):
        if self.board[:,j].sum() == player*LENGTH:
          self.winner = player
          self.ended = True
          return True

    # check diagonals
    for player in (self.x, self.o):
      # top-left -> bottom-right diagonal
      if self.board.trace() == player*LENGTH:
        self.winner = player
        self.ended = True
        return True
      # top-right -> bottom-left diagonal
      if np.fliplr(self.board).trace() == player*LENGTH:
        self.winner = player
        self.ended = True
        return True

    # check if draw
    if np.all((self.board == 0) == False):
      # winner stays None
      self.winner = 2
      self.ended = True
      return True

    # game is not over
    self.winner = None
    return False

  def is_draw(self):
    return self.ended and self.winner is 2

  def draw_board(self):
      #prints the tic tac toe board
    for i in range(LENGTH):
      print("-------------")
      for j in range(LENGTH):
        print("  ", end="")
        if self.board[i,j] == self.x:
          print("x ", end="")
        elif self.board[i,j] == self.o:
          print("o ", end="")
        else:
          print("  ", end="")
      print("")
    print("-------------")


#This class represents our bot
class Agent:
  def __init__(self, eps=0.1, alpha=0.5):
    self.eps = eps # probability of choosing random action instead of greedy
    self.alpha = alpha # learning rate
    self.verbose = False
    self.state_history = []
  
  def setV(self, V):
    self.V = V

  def set_symbol(self, sym):
    self.sym = sym

  def set_verbose(self, v):
    # if true, will print values for each position on the board
    self.verbose = v

  def reset_history(self):
    self.state_history = []

  def take_action(self, env):
    # choose an action based on epsilon-greedy strategy
    r = np.random.rand()
    best_state = None
    if r < self.eps:
      # take a random action
      if self.verbose:
        print("Taking a random action")

      possible_moves = []
      for i in range(LENGTH):
        for j in range(LENGTH):
          if env.is_empty(i, j):
            possible_moves.append((i, j))
      idx = np.random.choice(len(possible_moves))
      next_move = possible_moves[idx]
    else:
      # choose the best action based on current values of states
      pos2value = {} # for debugging
      next_move = None
      best_value = -1
      for i in range(LENGTH):
        for j in range(LENGTH):
          if env.is_empty(i, j):
            env.board[i,j] = self.sym
            state = env.get_state()
            env.board[i,j] = 0 
            pos2value[(i,j)] = self.V[state]
            if self.V[state] > best_value:
              best_value = self.V[state]
              best_state = state
              next_move = (i, j)

      # if verbose, draw the board w/ the values
      if self.verbose:
        print("Taking a greedy action")
        for i in range(LENGTH):
          print("------------------")
          for j in range(LENGTH):
            if env.is_empty(i, j):
              # print the value
              print(" %.2f|" % pos2value[(i,j)], end="")
            else:
              print("  ", end="")
              if env.board[i,j] == env.x:
                print("x  |", end="")
              elif env.board[i,j] == env.o:
                print("o  |", end="")
              else:
                print("   |", end="")
          print("")
        print("------------------")

    # make the move
    env.board[next_move[0], next_move[1]] = self.sym

  def update_state_history(self, s):
    '''
    cannot put this in take_action, because take_action only happens
    once every other iteration for each player
    state history needs to be updated every iteration
    '''
    self.state_history.append(s)

  def update(self, env):
    # we want to BACKTRACK over the states, so that:
    # V(prev_state) = V(prev_state) + alpha*(V(next_state) - V(prev_state))
    # where V(next_state) = reward if it's the most current state
    #
    # ONLY do this at the end of an episode
    reward = env.reward(self.sym)
    target = reward
    for prev in reversed(self.state_history):
      value = self.V[prev] + self.alpha*(target - self.V[prev])
      self.V[prev] = value
      target = value
    self.reset_history()    
    

class Human:
  def __init__(self):
    pass

  def set_symbol(self, sym):
    self.sym = sym

  def take_action(self, env):
    while True:
      # break if we make a legal move
      move = input("Enter coordinates i,j for your next move (i,j=0..2): ")    
      i, j = move.split(',')
      i = int(i)
      j = int(j)
      #error handling for invalid inputs
      while((i<0 or i>2) and (j<0 or j>2)):
          print("Please Enter Valid co-ordinates")
          print(" ")
          move = input("Enter coordinates i,j for your next move (i,j=0..2): ")
          i, j = move.split(',')
          i = int(i)
          j = int(j)
      if env.is_empty(i, j):
        env.board[i,j] = self.sym
        break

  def update(self, env):
    pass

  def update_state_history(self, s):
    pass


'''
recursive function that will return all possible states (as ints) and who 
the corresponding winner is for those states (if any) 
(i, j) refers to the next cell on the board to permute
'''
def get_state_hash_and_winner(env, i=0, j=0):
  results = []

  for v in (0, env.x, env.o):
    env.board[i,j] = v # if empty board it should already be 0
    if j == 2:
      # j goes back to 0, increase i, unless i = 2, then we are done
      if i == 2:
        # the board is full, collect results and return
        state = env.get_state()
        ended = env.game_over(force_recalculate=True)
        winner = env.winner
        results.append((state, winner, ended))
      else:
        results += get_state_hash_and_winner(env, i + 1, 0)
    else:
      # increment j, i stays the same
      results += get_state_hash_and_winner(env, i, j + 1)

  return results




def initialV_x(env, state_winner_triples):
    '''
   initialize state values as follows
   if x wins, V(s) = 1
   if x loses or draw, V(s) = 0
   otherwise, V(s) = 0.5
   '''
    V = np.zeros(env.num_states)
    for state, winner, ended in state_winner_triples:
        if ended:
          if winner == env.x:
              v = 1
          elif winner == 2:
              v = 0.8
          else:
              v=0
        else:
          v = 0.4
        V[state] = v
        return V


def initialV_o(env, state_winner_triples):
  #similar to initialV_x
  V = np.zeros(env.num_states)
  for state, winner, ended in state_winner_triples:
    if ended:
      if winner == env.x:
          v = 1
      elif winner == 2:
          v = 0.8
      else:
          v=0
    else:
      v = 0.4
    V[state] = v
  return V


def play_game(p1, p2, env, draw=False):
  # loops until the game is over
  current_player = None
  while not env.game_over():
    # alternating between two players
    # p1 always starts first
    if current_player == p1:
      current_player = p2
    else:
      current_player = p1

    # draw the board before the user who wants to see it makes a move
    if draw:
      if draw == 1 and current_player == p1:
        env.draw_board()
      if draw == 2 and current_player == p2:
        env.draw_board()

    # current player makes a move
    current_player.take_action(env)

    # update state histories
    state = env.get_state()
    p1.update_state_history(state)
    p2.update_state_history(state)

  if draw:
    env.draw_board()

  # do the value function update
  p1.update(env)
  p2.update(env)


if __name__ == '__main__':
  # train the agent
  p1 = Agent()
  p2 = Agent()

  # set initial V for p1 and p2
  env = Environment()
  state_winner_triples = get_state_hash_and_winner(env)


  Vx = initialV_x(env, state_winner_triples)
  p1.setV(Vx)
  Vo = initialV_o(env, state_winner_triples)
  p2.setV(Vo)

  # give each player their symbol
  p1.set_symbol(env.x)
  p2.set_symbol(env.o)

  T = 50000  #do 50000 iteration of agent vs agent game 
  for t in range(T):
    if t % 1000 == 0:
      print("%d games played" %t)
    play_game(p1, p2, Environment())

  # play human vs. agent
  human = Human()
  
  
  while True:
    turn = input("Do you want to play first or second? [1/2] : ")
    if turn == '1':
        human.set_symbol(env.x)        
        p2.set_verbose(True)
        play_game(human, p2, Environment(), draw=1)
        
        
    elif turn == '2':
        human.set_symbol(env.o)        
        p1.set_verbose(True)
        play_game(p1, human, Environment(), draw=2)
        
        
    else:
        print("Please Enter 1 or 2")
    
    answer = input("Play again? [Y/n] : ")
    if answer and answer.lower()[0] == 'n':
        break    
