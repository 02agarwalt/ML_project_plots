import numpy as np
import plotly as py
import plotly.offline as offline
import plotly.graph_objs as go

x_series = np.array(['No Correction', 'Quad Only', 'Small', 'Mid', 'Large'])
y_ranked_des = np.array([0.7698, 0.9118, 0.8993, 0.9298, 0.8066])
y_unranked_des = np.array([0.6842, 0.7623, 0.8280, 0.9230, 0.8066])

trace1 = go.Scatter(x=x_series, y=y_ranked_des, mode='markers', name='ranked', marker=dict(size=30))
trace2 = go.Scatter(x=x_series, y=y_unranked_des, mode='markers', name='unranked', marker=dict(size=30))

data = [trace1, trace2]

layout = dict(title="Comparison of Different Processing Pipelines for BNU1 Dataset", yaxis = dict(title="Discriminability"), xaxis = dict(title="Nuisance Correction Method in Pipeline"), font=dict(size=30))

fig = dict(data=data, layout=layout)

offline.plot(fig, "/mnr_plot.png")
