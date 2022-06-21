
import sfztest_1agent as model

model.EPOCH = 5000
model.decayed_step = 1500  # 1500
model.ep_mode = 1
model.CONVCEIL = 20
model.CONVBOUND = 0
model.N2 = 2
model.CLUSTER = 200
model.SHOW_REV_GRAPH = 1


m = model.sfz_dynamic_model(7706, dynamicGraph=0, verbose=0)
g1 = m.Generator()
for x in g1:
    pass
m.TextResult()
m.SavePickle()
