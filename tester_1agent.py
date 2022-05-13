
import sfztest_dynamic as model

model.EPOCH = 5000
model.decayed_step = 1500  # 1500
model.ep_mode = 0
model.CONVCEIL = 100
model.CONVBOUND = 5
model.N = 10
model.CLUSTER = 500
model.SHOW_REV_GRAPH = 1


m = model.sfz_dynamic_model(44, agentId=0, dynamicGraph=0, verbose=0)
g1 = m.Generator()
for x in g1:
    pass
m.TextResult()
m.SavePickle()
