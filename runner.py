
from qae import generate_random_dataset
from sspace import QCkt
from yacs.config import CfgNode as CN
from pprint import pprint

# add additional optimizers here 

# AVAILABLE_OPTIMIZERS = {
#     "rs": RandomSearchOptimizer
# }

def load_config(file):
    '''load and return a cfgnode object'''
    cfg = CN()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(file)
    return cfg


def run_nas_search(config):
    '''NAS search for different methods '''
    if config.optimizer == 'rs':
        random_search_optimizer(config)
    elif config.optimizer == 're':
        regularized_evolution_search(config)
    
    else:
        print("Invalid optimizer value found")
        raise NotImplementedError()



def random_search_optimizer(config):
    sampler = QCkt(config.ae.nlatent + config.ae.ntrash)
    data = generate_random_dataset(config, sampler,
                                    population_size=config.opt_params.population_size)

    top_5_fids = sorted([m.fidelity for m in data])[:min(5, len(data))]
    print(f"Random search, population size: {config.opt_params.population_size}")
    print("Top 5 fidelities: ", top_5_fids)
    return data


def regularized_evolution_search(config):
    pass




if __name__ == '__main__':
    cfg = load_config('digits_config.yml')
    
    run_nas_search(config=cfg)
