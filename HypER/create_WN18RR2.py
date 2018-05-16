import os

def load_data(data_dir, data_type="train"):
    with open("%s%s.txt" % (data_dir, data_type), "r") as f:
        data = f.read().strip().split("\n")
        data = [i.split() for i in data]
    return data

def get_entities(data):
    entities = set([d[0] for d in data]+[d[2] for d in data])
    return entities

def get_relations(data):
    relations = set([d[1] for d in data])
    return relations

def create_WN18RR2(data, train_entities):
	data = [i for i in data if i[0] in train_entities and i[2] in train_entities]
	return data

def produce_output(data):
	data = "\n".join(["\t".join(i) for i in data])
	return data

data_dir = "data/WN18RR/"
train_data = load_data(data_dir, "train")
valid_data = load_data(data_dir, "valid")
test_data = load_data(data_dir, "test")
train_entities = get_entities(train_data)
valid_data_wn18rr2 = create_WN18RR2(valid_data, train_entities)
test_data_wn18rr2 = create_WN18RR2(test_data, train_entities)

print("Number of entities:")
print(len(get_entities(train_data)|get_entities(valid_data_wn18rr2)|get_entities(test_data_wn18rr2)))
print("Number of relations:")
print(len(get_relations(train_data)|get_relations(valid_data_wn18rr2)|get_relations(test_data_wn18rr2)))


data_dir_wn18rr2 = "convE/data/WN18RR2/"
if not os.path.exists(data_dir_wn18rr2):
    os.makedirs(data_dir_wn18rr2)
with open("%strain.txt" % (data_dir_wn18rr2), "w") as f:
	f.write(produce_output(train_data))
with open("%svalid.txt" % (data_dir_wn18rr2), "w") as f:
	f.write(produce_output(valid_data_wn18rr2))
with open("%stest.txt" % (data_dir_wn18rr2), "w") as f:
	f.write(produce_output(test_data_wn18rr2))





