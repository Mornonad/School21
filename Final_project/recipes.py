import pandas as pd
import joblib
from joblib import dump, load
import numpy as np
import warnings
warnings.simplefilter(action='ignore')

class Forecast:
    """
    Предсказание рейтинга блюда или его класса
    """
    def __init__(self, list_of_ingredients):
        self.list_of_ingredients = list_of_ingredients
        self.df = pd.read_csv('data/result.csv')
        self.df = self.df.drop(columns=['title', 'url', 'rating', 'rating_group'])
        self.ing = self.df.columns

    def check_input(self):
        not_found = []
        for i in self.list_of_ingredients:
            if i not in list(self.ing):
                print(f'\nИнгредиент {i} не найден')
                not_found.append(i)
                new_ing = input('Добавьте новый ингредиент: ')
                self.list_of_ingredients.append(new_ing)
        for value in not_found:
            self.list_of_ingredients.remove(value)
        return self.list_of_ingredients
  
    def preprocess(self):
        """
        Этот метод преобразует список ингредиентов в структуры данных, 
        которые используются в алгоритмах машинного обучения, чтобы сделать предсказание.
        """
        df = pd.read_csv('data/result.csv')
        df = df.drop(columns=['title','url', 'rating', 'rating_group'])
        vector = pd.DataFrame(np.zeros(df.shape), columns=df.columns)
        if set(self.list_of_ingredients).issubset(df.columns):
            vector[self.list_of_ingredients] = 1
        
        return vector
        
    def predict_rating_category(self):
        """
        Этот метод возращает рейтинговую категорию для списка ингредиентов, используя классификационную модель, 
        которая была обучена заранее. Помимо самого рейтинга, метод возвращает также и текст, 
        который дает интерпретацию этой категории и дает рекомендации, как в примере выше.
        """
        data = self.preprocess()
        model = joblib.load('data/best_model')
        rating_cat = model.predict(data)       
        print('I. НАШ ПРОГНОЗ:\n')
        
        if rating_cat[0] == 'bad':
            return 'Невкусное. Хоть конкретно вам может быть и понравится блюдо из этих ингредиентов, но, на наш взгляд, это плохая идея – готовить блюдо из них. Хотели предупредить.\n'
        elif rating_cat[0] == 'so-so':
            return 'Нормальное. Но если Вы гурман, стоит поискать что-то другое.\n'
        elif rating_cat[0] == 'great':
            return 'Вкусное. На наш скромный взгляд из этих ингредиентов получится изысканное блюдо.\n'
        else:
            return 'По данным ингредиентам ничего не нашлось. Попробуйте ввести что-то другое.\n'

class NutritionFacts:
    """
    Выдает информацию о пищевой ценности ингредиентов.
    """
    def __init__(self, list_of_ingredients):
        self.list_of_ingredients = list_of_ingredients
        self.nutrients = pd.read_csv('data/Nutrients.csv')
        
    def retrieve(self):
        """
        Этот метод получает всю имеющуюся информацию о пищевой ценности из файла с заранее 
        собранной информацией по заданным ингредиентам. 
        Он возвращает ее в том виде, который вам кажется наиболее удобным и подходящим.
        """

        facts =pd.DataFrame(columns=self.nutrients.columns)
        
        for i in self.list_of_ingredients:
            nutr_el =self.nutrients.loc[self.nutrients.ingredient == i]
            facts = pd.concat([facts, nutr_el], ignore_index=True)          
        facts.drop(columns='Unnamed: 0',inplace=True)
        
        return facts

    def filter(self, must_nutrients, n):
        """
        Этот метод отбирает из всей информации о пищевой ценности только те нутриенты, 
        которые были заданы в must_nutrients (пример в PDF-файле ниже), 
        а также топ-n нутриентов с наибольшим значением дневной нормы потребления для заданного ингредиента. 
        Он возвращает текст, отформатированный как в примере выше.
        """
        
        facts = self.retrieve()
        info = {}
        print(f'II. ПИЩЕВАЯ ЦЕННОСТЬ \n')        
        
        for i in self.list_of_ingredients:
            per_ingred = facts.loc[facts['ingredient'] == i].sort_values('value',ascending=False)
            print(i.capitalize())
                
            for j in range(len(must_nutrients)):
                if j < n:
                    if must_nutrients[j] in per_ingred['nutrientName'].tolist():
                        nutrients = per_ingred.loc[per_ingred['nutrientName']== must_nutrients[j]]
                        value =per_ingred.loc[per_ingred['nutrientName']== must_nutrients[j]]['value']
                        print(f'{must_nutrients[j]} - {float(value)} % of daily value')
                    else:
                        print(f'Нутриент {must_nutrients[j]} для данного ингредиента не найден')
                else:
                    break

            print('\n')

class SimilarRecipes:
    """
    Рекомендация похожих рецептов с дополнительной информацией
    """
    def __init__(self, list_of_ingredients):
        self.list_of_ingredients = list_of_ingredients
        self.recipes = pd.read_csv('data/result.csv')
        self.ingredients = list(self.recipes.drop(columns=['title', 'url','rating','rating_group']).columns)
        
    def find_all(self):
        """
        Этот метод возвращает список индексов рецептов, которые содержат заданный список ингредиентов. 
        Если нет ни одного рецепта, содержащего все эти ингредиенты, 
        то сделайте обработку ошибки, чтобы программа не ломалась.
        """

        if set(self.list_of_ingredients).issubset(self.recipes.columns):
            recipes = self.recipes[self.list_of_ingredients]
            results = recipes.sum(axis=1) == len(self.list_of_ingredients)
            indexes = recipes.loc[results].index.tolist()
            return indexes
        else:
            indexes = []
            return indexes

    
    def top_similar(self, n):
        """
        Этот метод возвращает текст, форматированный как в примере выше: с заголовком, рейтингом и URL. 
        Чтобы это сделать, он вначале находит топ-n наиболее похожих рецептов 
        с точки зрения количества дополнительных ингредиентов, 
        которые потребуются в этих рецептах. Наиболее похожим будет тот, в котором не требуется никаких других ингредиентов. 
        Далее идет тот, у которого появляется 1 доп. ингредиент. Далее – 2. 
        Если рецепт нуждается в более, чем 5 доп. ингредиентах, то такой рецепт не выводится.
        """
        print(f'III. ТОП-{n} ПОХОДИХ РЕЦЕПТА \n')
        
        indexes = self.find_all()
        
        if len(indexes) != 0:
            final_results = []

            for i in indexes:
                if self.recipes.iloc[i].loc[self.ingredients].sum() - len(self.list_of_ingredients) < 5:
                    if self.recipes.iloc[i].loc[self.ingredients].sum() == len(self.list_of_ingredients):
                        final_results.append({'name':self.recipes.iloc[i].title, 
                                              'rating': self.recipes.iloc[i].rating, 
                                              'url': self.recipes.iloc[i].url})
                    elif self.recipes.iloc[i].loc[self.ingredients].sum() - 1 == len(self.list_of_ingredients):
                        final_results.append({'name':self.recipes.iloc[i].title, 
                                              'rating': self.recipes.iloc[i].rating, 
                                              'url': self.recipes.iloc[i].url})
                    elif self.recipes.iloc[i].loc[self.ingredients].sum() - 2 == len(self.list_of_ingredients):
                        final_results.append({'name':self.recipes.iloc[i].title, 
                                              'rating': self.recipes.iloc[i].rating, 
                                              'url': self.recipes.iloc[i].url})
                    elif self.recipes.iloc[i].loc[self.ingredients].sum() - 3 == len(self.list_of_ingredients):
                        final_results.append({'name':self.recipes.iloc[i].title, 
                                              'rating': self.recipes.iloc[i].rating, 
                                              'url': self.recipes.iloc[i].url})
                    elif self.recipes.iloc[i].loc[self.ingredients].sum() - 4 == len(self.list_of_ingredients):
                        final_results.append({'name':self.recipes.iloc[i].title, 
                                              'rating': self.recipes.iloc[i].rating, 
                                              'url': self.recipes.iloc[i].url})
            for j in final_results[:n]:
                print(j.get('name'), j.get('rating'), j.get('url'))

        else:
            print('Похожих рецептов не найдено.\n')
                
class Recommend_Menu():
    
    def __init__(self):
        self.data = pd.read_csv('data/result.csv')
        self.full_recipes = pd.read_json('data/full_format_recipes.json', orient='columns')
        self.full_recipes = self.full_recipes[self.full_recipes['categories'].notna()]
        self.full_recipes = self.full_recipes.loc[self.full_recipes['title'].isin(list(self.data['title']))]
        self.meal = [['Breakfast','ЗАВТРАК'], ['Lunch','ОБЕД'], ['Dinner','УЖИН'], ['Dessert','ДЕСЕРТ']]
        
    def recommend(self):

        for i in self.meal:
            print(f'{i[1]}\n')
            info = self.full_recipes[self.full_recipes.categories.map(set([i[0]]).issubset)].sample(n=1)

            name = info.title.tolist()[0]

            info_url = self.data.loc[self.data.title ==name]
            print(F'{name} (рейтинг: {info.rating.tolist()[0]})\n')
            print('Ингредиенты: \n')
            info_ings = info_url.drop(columns=['title', 'url', 'rating', 'rating_group']).T
            info_ings = info_ings.reset_index().rename(columns={'index':'ings'})
            
            ings = info_ings.loc[info_ings[info_ings.columns[1]] == 1].index.tolist()
            ings = info_ings.iloc[ings]['ings'].tolist()

            for j in ings:
                print(f'-{j}')
            print('\n') 
            print('Nutrients: \n')
            print(f'- calories: {round(float(info.calories)/2500* 100, 2)} %\n- protein: {round(float(info.protein) / 150 * 100, 2)} %\
            \n- fat: {round(float(info.fat) / 178 * 100.2)} %\n- sodium:{round(float(info.sodium) / 2300 * 100,2)} %\n')
            print(f'Ссылка на рецепт: {info_url.url.tolist()[0]}\n')