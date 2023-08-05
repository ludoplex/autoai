# Copyright 2021 BlobCity, Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This Python File Consists of Functions to perform Automatic feature Selection 

"""
import os,cv2
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold,SelectKBest,f_regression,f_classif
from sklearn.preprocessing import  MinMaxScaler
from statistics import mean
from blobcity.utils.Cleaner import dataCleaner
class AutoFeatureSelection: 

    def dropHighCorrelationFeatures(self):
        """
        param1: pandas DataFrame

        return: pandas DataFrame

        Function calculates Mutual Correlation of the passed dataframe,
        and drop one of the feature which are highly correlated to each other,
        """
        cor_matrix = self.corr()
        upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        return self.drop(to_drop, axis=1) if to_drop!=[] else self
            
    def dropConstantFeatures(self):
        """
        param1: pandas DataFrame
        return: pandas DataFrame
        
        Funciton drops column with low variance/constant values in the field.
        VarianceThreshold function is utilized to filter out such columns from the DataFrame
        and once all such columns are dropped if exists return the dataframe.
        """
        cols = self.columns
        constant_filter = VarianceThreshold(threshold=0).fit(self)
        constcols=[col for col in cols if col not in cols[constant_filter.get_support()]]
        if (constcols!=[]):
            self.drop(constcols, axis=1, inplace=True)
        return self

    def MainScore(self, dict_class):

        """
        param1: dictionary - feature importance scores 
        param2: Class Object - Dictionary class

        return: dictionary 

        Function calculate and filter dictionary on the basis on the existence of Categorical feature(String type features).
        first the function check whether the dataset had any String categorical feature using previously stored Boolean Class variable .
        If String Categorical feature  exist return aggregate score for the actually features and return it.
        else return passed feature score.
        """
        #average on categorical features
        if not dict_class.ObjectExist:
            return self
        objList=dict_class.ObjectList
        for i in objList:
            self[i] = mean(
                list(
                    dict(filter(lambda item: i in item[0], self.items())).values()
                )
            )
        return {
            key: val
            for key, val in self.items()
            if all(ele + "_" not in key for ele in objList)
        }

    #feature importance calculator
    def get_feature_importance(self, Y, score_func, dict_class):
        """
        param1: pandas DataFrame X Features
        param2: pandas Series/Dataframe target dataset
        param3: Sklearn.feature_selection function 
        param4: Class Object of Dictionary Class

        return: pandas DataFrame

        Working:

        Function  Selects Feature on the basis of feature importance using f_classif or f_regression function.
        Feature importance score generated using SelectKBest Function is then Scaled in range of 0 to 1, using MinMaxScaler function
        To manage feature from one hot encoding if any categorical feature exists MainScore function returns an average/mean score for the appropirate feature.
        if the dataframe has less then equal to 2 features return orignal dataframe. else return a short listed dataframe on the basis of 
        categorical features.
        """
        if self.shape[1] < 3:
            dict_class.feature_importance=None
            return self
        else:
            fit = SelectKBest(score_func=score_func, k=self.shape[1]).fit(self, Y)
            dfscores,dfcolumns = pd.DataFrame(fit.scores_), pd.DataFrame(self.columns)
            df = pd.concat([dfcolumns,dfscores],axis=1)
            df.columns = ['features','Score']
            df['Score']=MinMaxScaler().fit_transform(np.array(df['Score']).reshape(-1,1))
            main_score=AutoFeatureSelection.MainScore(dict(df.values),dict_class)
            return AutoFeatureSelection.GetAbsoluteList(
                main_score, self, dict(df.values), dict_class
            )
    
    def GetAbsoluteList(self, dataframe, impmain, dict_class):

        """
        param1: Dictionary
        param2: pandas.DataFrame
        param3: Dictionary

        return: pandas.DataFrame
        """
        keylist=[]
        imp_dict={}
        for key, value in self.items():
            if value < 0.01: 
                keylist.extend(key_2 for key_2 in impmain.keys() if key in key_2)
            else: imp_dict[key]=value

        result_df=dataframe.drop(keylist,axis=1)
        dict_class.feature_importance=imp_dict
        return result_df

    def FeatureSelection(self, target, dict_class, disable_colinearity):
        """
        param1: pandas DataFrame
        param2: target column name
        param3: Class object
        
        return : pandas dataframe

        Function starting with performing data cleaning process of the data by calling dataCleaner funciton.
        On the Basis of problem type either Classification or Regression assigns scoring function for feature selection.
        perform a subset for feature set and target set.
        Pass the Feature set/independent features through feature selection process has follow:
            1. Droping Constant Features
            2. Droping Highly Correlated features
            3. Droping Columns on basis of Feature Importance Criteria.
        and finally return List of features to utilize ahead for processing and model training.
        """
        df = dataCleaner(
            self, self.drop(target, axis=1).columns.to_list(), target, dict_class
        )
        score_func=f_classif if(dict_class.getdict()['problem']["type"]=='Classification') else f_regression
        X=df.drop(target,axis=1)
        Y=df[target]

        X=AutoFeatureSelection.dropConstantFeatures(X)
        X=AutoFeatureSelection.dropHighCorrelationFeatures(X) if not disable_colinearity else X
        X=AutoFeatureSelection.get_feature_importance(X,Y,score_func,dict_class)
        featureList=AutoFeatureSelection.getOriginalFeatures(X.columns.to_list(),dict_class)
        dict_class.addKeyValue('features',{'X_values':featureList,'Y_values':target})
        return featureList

    def getOriginalFeatures(self, dict_class):
        """
        param1: List
        param2: Class object
        
        return: List

        Function check whether Object type data exists using Class Variable.
        if exists shorts/filters list of feature on the basis of feature importance list.
        and return filtered List of features
        """
        if not dict_class.ObjectExist:
            return self
        res,res2= [],[]#List
        res.extend(
            val
            for val in self
            if all(ele + "_" not in val for ele in dict_class.ObjectList)
        )
        res += dict_class.ObjectList
        res2.extend(v for v in res if all(v not in ele for ele in featureList))
        return [i for i in res if i not in res2]

    def image_processing(self, targets, resize, dict_class):
        """
        param1: String
        param2: List
        param3: Class object
        return: pandas.DataFrame

        Function return dataframe object which consist of image data and the target.
        """
        training_data, label_mapping = AutoFeatureSelection.create_training_data(
            self, targets, resize
        )
        dict_class.original_label=label_mapping
        original_shape = [len(training_data), *list(training_data[0][0].shape)]
        dict_class.original_shape=original_shape
        dict_class.addKeyValue('cleaning',{"resize":resize})
        return pd.DataFrame(training_data, columns=['image', 'label'])

    def create_training_data(self, target, resize):
        """
        param1: String
        param2: List
        return: Tuple : (List,dict)

        Function read each image from the provided source and creates a training data whether each image data is mapped with its target label.
        """
        training_data=[]
        label_mapping={}
        for category in target:
            path = os.path.join(self, category)
            class_num=target.index(category)
            label_mapping[class_num]=category
            for img in os.listdir(path):
                try:
                    img_array=cv2.imread(os.path.join(path,img))
                    img_array=cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB)
                    new_array=cv2.resize(img_array,(resize,resize))
                    training_data.append([new_array,class_num])
                except Exception as e:
                    pass
        return (training_data,label_mapping)

    def get_reshaped_image(self):
        """
        param1:numpy.array
        return: tuple :(numpy.array,numpy.array)

        Function return required feature and target in a flatten image and label for training 
        """
        lenofimage = len(self)
        X, y = [], []
        for categories, label in self:
            X.append(categories)
            y.append(label)
        X = np.array(X).reshape(lenofimage,-1)
        y = np.array(y)
        return (X,y)