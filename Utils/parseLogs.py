import ast
import re
import matplotlib.pyplot as plt

def plotFigure(x,y, name,path):
    path = "/home/mgroepl/plots/"
    plt.figure()  # Start a new figure
    plt.plot(x, y, marker='o')
    plt.title(name)
    plt.xlabel('epoch')
    plt.ylabel('training ' + name)
    plt.grid(True)
    plt.savefig(path+ name+".png")
    plt.close()


def extract_dictionaries_from_file(file_path):
    dictionaries = []

    # Regex to find something that *looks* like a dict: { ... }
    dict_pattern = re.compile(r'\{.*?\}')

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            matches = dict_pattern.findall(line)
            for match in matches:
                try:
                    parsed = ast.literal_eval(match)
                    if isinstance(parsed, dict):
                        dictionaries.append(parsed)
                except (SyntaxError, ValueError):
                    continue  # Not a valid dict

    return dictionaries

##set the path to the train log you want to plot  and where you want to save the plots
file_path = '/path/to/log/log.out'
save_path = "/path/where/to/save/plots/
dicts = extract_dictionaries_from_file(file_path)

epochs = []
loss = []
grad_norm = []
learning_rate = []
num_tokens = []
completion_length = []
reward_std= []
kl = []
reward = []

for dict in dicts:
    try:
        epochs.append(dict["epoch"])
        loss.append(dict["loss"])
        grad_norm.append(dict["grad_norm"])
        learning_rate.append(dict["learning_rate"])
        num_tokens.append(dict["num_tokens"])
        completion_length.append(dict["completion_length"])
        reward_std.append(dict["reward_std"])
        kl.append(dict["kl"])
        reward.append(dict["reward"])

    except KeyError:
        continue

plotFigure(epochs,loss,"loss",save_path)
plotFigure(epochs,grad_norm,"grad_norm",save_path)
plotFigure(epochs,learning_rate,"learning_rate",save_path)
plotFigure(epochs,num_tokens,"num_tokens",save_path)
plotFigure(epochs,completion_length,"completion_lenght",save_path)
plotFigure(epochs,reward_std,"reward_std",save_path)
plotFigure(epochs,kl,"kl",save_path)
plotFigure(epochs,reward,"reward",save_path)
