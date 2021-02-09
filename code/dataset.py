import pandas as pd

parent_dir = '../data/UCI/'
dataset_paths = ['zoo/zoo.data',
                 'votes/house-votes-84.data',
                 'breast_cancer/breast-cancer-wisconsin.data',
                 'mushroom/agaricus-lepiota.data',
                 'balance_scale/balance-scale.data',
                 'car/car.data',
                 'chess/kr-vs-kp.data',
                 'hayes_roth/hayes-roth.data',
                 'nursery/nursery.data',     
                 'soybean/soybean-small.data']

columns = [['animal name', 'hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic',
            'predator', 'toothed', 'backbone', 'breathes', 'venomous', 'fins', 'legs', 'tail',
            'domestic', 'catsize', 'class'],
           ['class', 'handicapped_inf', 'water_proj_cost', 'adpt_budget_res',
            'phy_fee_freeze', 'el_sal_aid', 'sch_religious_grps', 'anti_sat_ban',
            'aid-to-nicaraguan-contras', 'mx-missile', 'immigration',
            'synfuels-corporation-cutback', 'education-spending', 'superfund-right-to-sue',
            'crime', 'duty-free-exports', 'export-administration-act-south-africa'],
           ['sample_code_no', 'clump_thickness', 'uni_cell_size', 'uni_cell_shape',
            'mar_adhesion', 'single_epithelial_cell_size', 'bare_nuclei', 'bland_chromatin',
            'normal_nucleoli', 'mitoses', 'class'],
           ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises?', 'odor',
            'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape',
            'stalk-root', 'stalk-sur-abv-ring', 'stalk-sur-below-ring', 'stalk-col-abv-ring',
                                                'stalk-col-below-ring', 'veil-type', 'veil-color', 'ring-nu', 'ring-type',
                                                'spore-color', 'population', 'habitat'],
           ['class', 'left_weight', 'left_dist', 'right_weight', 'right_dist'],
           ['buying', 'maint', 'doors', 'persons','lug_boot', 'safety', 'class'],
           ['mv{}'.format(i) for i in range(36)] + ['class'],
           ['name', 'hobby', 'age', 'edu_lvl', 'marital_status', 'class'],
           ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'class'],
           ['date', 'stand', 'precip', 'temp', 'hail','hist', 'damaged', 'severity', 'seed-tmt', 'germination', 'growth',
           'leaves', 'halo', 'margs', 'size', 'shread', 'malf', 'mild', 'stesm', 'lodging', 'cankers', 'lesion', 'bodies',
           'decay', 'mycelium', 'discolor', 'sclerotia', 'pods', 'spots', 'seed', 'mold', 'seed-discolor', 'seedsize', 
           'shriveling', 'roots', 'class']]

def get_datasets():
    datasets = []
    for path, headers in zip(dataset_paths, columns):
        datasets.append(pd.read_csv(parent_dir+path, names=headers))
    return datasets
