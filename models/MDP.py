# import numpy as np
# import matplotlib.pyplot as plt
#
# class MDP():
#     # MDP Abstract class that defines the basic properties and methods of a
#     # problem (number of states and actions, simulation and plotting, ...).
#
#
#         initplot(obj)
#
#         updateplot(obj,state)
#
#
#     def get_state_idx(obj = None,s = None):
#         # Return a vector with the indices of the the states s wrt their
#     # tabular representation, i.e., obj.allstates(idx,:) = s
#     # ONLY FOR MDP WITH DISCRETE STATES
#     # S must be [dS x N].
#         __,idx = ismember(np.transpose(s),obj.allstates,'rows')
#         return idx
#
#
#     def initstate(obj = None,n = None):
#         # Return N initial states.
#         if len(varargin) == 1:
#             n = 1
#
#         state = obj.init(n)
#         if obj.realtimeplot:
#             obj.showplot
#             obj.updateplot(state)
#
#         return state
#
#
#     def simulator(obj = None,state = None,action = None):
#         # Defines the state transition function.
#         action = obj.parse(action)
#         nextstate = obj.transition(state,action)
#         reward = obj.reward(state,action,nextstate)
#         absorb = obj.isterminal(nextstate)
#         if obj.realtimeplot:
#             obj.updateplot(nextstate)
#
#         return nextstate,reward,absorb
#
#
#     def showplot(obj = None):
#         # Initializes the plotting procedure.
#         obj.realtimeplot = 1
#         if len(obj.handleEnv)==0:
#             obj.initplot()
#
#         if not isvalid(obj.handleEnv) :
#             obj.initplot()
#
#         return
#
#
#     def pauseplot(obj = None):
#         # Pauses the plotting procedure.
#         obj.realtimeplot = 0
#         return
#
#
#     def resumeplot(obj = None):
#         # Resumes the plotting procedure.
#         obj.realtimeplot = 1
#         return
#
#
#     def closeplot(obj = None):
#         # Closes the plots and stops the plotting procedure.
#         obj.realtimeplot = 0
#         try:
#             close_(obj.handleEnv)
#         finally:
#             pass
#
#         obj.handleEnv = []
#         obj.handleAgent = []
#         return
#
#
#     def plotepisode(obj = None,episode = None,pausetime = None):
#         # Plots the state of the MDP during an episode.
#         if len(varargin) == 2:
#             pausetime = 0.001
#
#         try:
#             close_(obj.handleEnv)
#         finally:
#             pass
#
#         obj.initplot()
#         obj.updateplot(episode.s(:,1))
#         for i in np.arange(1,episode.nexts.shape[2-1]+1).reshape(-1):
#             pause(pausetime)
#             obj.updateplot(episode.nexts(:,i))
#             plt.title(np.array(['Step ',num2str(i),',   Reward ',mat2str(np.transpose(episode.r(:,i))).replace(' ',', ')]))
#
#         return
#
#
#     def plot_trajectories(obj = None,policy = None,episodes = None,steps = None):
#         if len(varargin) < 4 or len(steps)==0:
#             steps = 50
#
#         if len(varargin) < 3 or len(episodes)==0:
#             episodes = 10
#
#         obj.closeplot
#         ds = collect_samples(obj,episodes,steps,policy)
#         obj.showplot
#         hold('all')
#         if len(obj.stateUB) == 2:
#             for i in np.arange(1,np.asarray(ds).size+1).reshape(-1):
#                 s = np.transpose(np.array([ds(i).s,ds(i).nexts(:,end())]))
#                 plt.plot(s(:,1),s(:,2),'o-')
#         else:
#             if len(obj.stateUB) == 3:
#                 for i in np.arange(1,np.asarray(ds).size+1).reshape(-1):
#                     s = np.transpose(np.array([ds(i).s,ds(i).nexts(:,end())]))
#                     plot3(s(:,1),s(:,2),s(:,3),'o-')
#             else:
#                 raise Exception('Cannot plot trajectories for more than 3 dimensions.')
#
#         return
#