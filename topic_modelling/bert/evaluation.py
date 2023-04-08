import pandas as pd
import sys
import os
import syspend
from utility import parse_config, custom_print
from preprocess_class import create_datasets
from BERTopic_model import BERTopic_model

###### Driver class
if __name__ =="__main__":
    curr_dir = os.getcwd()
    config_path = os.path.join(curr_dir, 'bert_topic_config.yml')
    config_file = parse_config(config_path)
    data_file = config_file['data_folder']
    home_folder = os.path.abspath(os.path.join(os.getcwd(),'../..'))
    data_df = pd.read_csv(os.path.join(home_folder,data_file))
    train, test = create_datasets(data_df)


    model_name = 'sentence-transformers-key-bigram'
    logging_path = os.path.join(curr_dir,config_file['log_path'],f'{model_name}.log')
    model = BERTopic_model()
    model.load_model(model_name)
    logger = open(os.path.join(curr_dir, logging_path), 'a')
    ##Manually name each topic based on the top topic key words and also by inspecting samples from each topic
    labeled_train = pd.DataFrame({'Text': train.text,'Label':model.topic_model.topics_})
    for i in range(model.topic_model.get_topic_freq().shape[0] -1):
        custom_print(f'{model.topic_model.custom_labels_[i]}', logger=logger)
        custom_print(labeled_train.loc[labeled_train['Label']==i, 'Text'].sample(n=10), logger=logger)
    model.topic_model.set_topic_labels({
    -1: 'Outliers',
    0: 'Staple Food/Cooking Products',
    1: 'Coffee',
    2: 'Tea',
    3: 'Snacks',
    4: 'Dog Food',
    5: 'Baby Food/Canned Food',
    6: 'Delivery',
    7: 'Processed Food',
    8: 'Cat Food',
    9: 'Concentrated Syrup',
    10: 'Popcorn',
    11: 'Coconut Products',
    12: 'Protein Powder',
    13: 'Consumable Oil',
    14: 'Soup'})
    #generate csv file on test dataset for manual evaluation
    test_label = model.predict(test)
    labeled_test = pd.DataFrame({'Text': test.text,'Label':test_label[0]})
    lab = model.topic_model.get_topic_info().loc[:,['Topic', 'CustomName']]
    labeled_test = labeled_test.merge(lab, how='left', left_on='Label', right_on='Topic')
    #Randomly sample
    out=labeled_test.loc[labeled_test['Label']==-1].sample(n=8, random_state= 4263)   
    for i in range(model.topic_model.get_topic_freq().shape[0] -1):
        out = out.append(labeled_test.loc[labeled_test['Label']==i].sample(n=10, replace=False, random_state= 4263))
    out.drop(columns=['Label', 'Topic'], inplace=True)
    out.to_csv(f'{model_name}.csv', index=False)
    #Full dataset
    labeled_test.drop(columns=['Label'], inplace=True)
    labeled_test.to_csv(f'{model_name}_full.csv', index=False)
    model.topic_model.save(model_name)



    #Next model
    model_name = 'sentence-transformers-key-bigram-v2'
    logging_path = os.path.join(curr_dir,config_file['log_path'],f'{model_name}.log')
    model.load_model(model_name)
    logger = open(os.path.join(curr_dir, logging_path), 'a')

    ##Manually name each topic based on the top topic key words and also by inspecting samples from each topic
    labeled_train = pd.DataFrame({'Text': train.text,'Label':model.topic_model.topics_})
    for i in range(model.topic_model.get_topic_freq().shape[0] -1):
        custom_print(f'{model.topic_model.custom_labels_[i]}', logger=logger)
        custom_print(labeled_train.loc[labeled_train['Label']==i, 'Text'].sample(n=10), logger=logger)
    model.topic_model.set_topic_labels({
    -1: 'Outliers',
    0: 'Soft Drinks',
    1: 'Tea/Coffee',
    2: 'Dog Food',
    3: 'Sauces',
    4: 'Baking Products',
    5: 'Snack Bars',
    6: 'Chips',
    7: 'Keurig Coffee Products',
    8: 'Cereal',
    9: 'Noodle',
    10: 'Salty Food Products',
    11: 'Crackers',
    12: 'Cat Food',
    13: 'Coconut Products',
    14: 'Popcorn',
    15: 'Oil Products',
    16: 'Soup',
    17: 'Protein Powder',
    18: 'Peanut Butter'
    })
    #generate csv file on test dataset for manual evaluation
    test_label = model.predict(test)
    labeled_test = pd.DataFrame({'Text': test.text,'Label':test_label[0]})
    lab = model.topic_model.get_topic_info().loc[:,['Topic', 'CustomName']]
    labeled_test = labeled_test.merge(lab, how='left', left_on='Label', right_on='Topic')
    #Randomly sample
    out=labeled_test.loc[labeled_test['Label']==-1].sample(n=8, random_state= 4263)   
    for i in range(model.topic_model.get_topic_freq().shape[0] -1):
        out = out.append(labeled_test.loc[labeled_test['Label']==i].sample(n=10, replace=False, random_state= 4263))
    out.drop(columns=['Label', 'Topic'], inplace=True)
    out.to_csv(f'{model_name}.csv', index=False)
    #Full dataset
    labeled_test.drop(columns=['Label'], inplace=True)
    labeled_test.to_csv(f'{model_name}_full.csv', index=False)
    model.topic_model.save(model_name)	



    #Next model
    model_name = 'zero-shot'
    logging_path = os.path.join(curr_dir,config_file['log_path'],f'{model_name}.log')
    model.load_model(model_name)
    logger = open(os.path.join(curr_dir, logging_path), 'a')

    ##Merge topics with the same name but listed under different category
    labeled_train = pd.DataFrame({'Text': train.text,'Label':model.topic_model.topics_})
    for i in range(model.topic_model.get_topic_freq().shape[0] -1):
        custom_print(labeled_train.loc[labeled_train['Label']==i, 'Text'].sample(n=10), logger=logger)
    topics_to_merge= [[0,1,7,12,13,21,28], [2,14,22,8,20], [3,25,31],[5,23],[9,10,16,19],[6,11,15,24,26,17],
                  [-1,27,29,30]]
    model.topic_model.merge_topics(train.text, topics_to_merge)
    model.topic_model.set_topic_labels({-1: "Outliers", 0: "Drinks", 1: "Snacks", 2:"Healthy Alternatives",
                                    3:"Carbohydrates", 4:"Household", 5:"Sauce", 6:'Dogs', 7:"Cats"})

    test_label = model.predict(test)
    labeled_test = pd.DataFrame({'Text': test.text,'Label':test_label[0]})
    lab = model.topic_model.get_topic_info().loc[:,['Topic', 'CustomName']]
    labeled_test = labeled_test.merge(lab, how='left', left_on='Label', right_on='Topic')
    #Randomly sample
    out=labeled_test.loc[labeled_test['Label']==-1].sample(n=10, random_state= 4263)   
    for i in range(model.topic_model.get_topic_freq().shape[0] -1):
        out = out.append(labeled_test.loc[labeled_test['Label']==i].sample(n=10, replace=False, random_state= 4263))
    out.drop(columns=['Label', 'Topic'], inplace=True)
    out.to_csv(f'{model_name}.csv', index=False)

    labeled_test.drop(columns=['Label'], inplace=True)
    labeled_test.to_csv(f'{model_name}_full.csv', index=False)
    model.topic_model.save(model_name)

    #Did not evaluate the bert-based-uncased models as evaluation on based the topics generated and plot generated already shown it did not perform well
