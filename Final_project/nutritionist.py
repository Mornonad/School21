import pandas as pd
import recipes

def programm():

    list_of_ingredients = input('Введите не менее двух ингредиентов на английском языке через запятую:')
    list_of_ingredients= list_of_ingredients.lower().split(',')
    list_of_ingredients = [i.strip(' ') for i in list_of_ingredients]

    recipes.Forecast(list_of_ingredients).check_input()

    if len(list_of_ingredients) < 2:
        print('Пожалуйста, введите не менее двух ингредиентов')
    elif len(list_of_ingredients) >= 2:

        prediction = recipes.Forecast(list_of_ingredients).predict_rating_category()
        print(prediction)
        recipes.NutritionFacts(list_of_ingredients).filter(
            ['Protein', 'Total lipid (fat)', 'Calcium, Ca', 'Sodium, Na', 'Vitamin C', 'Cholesterol'], 6)
        recipes.SimilarRecipes(list_of_ingredients).top_similar(3)

        flag = input(f'\nПредложить меню на день? y/n:')
        if flag == 'n':
            print('Хорошо')
        elif flag == 'y':
            recommendation = recipes.Recommend_Menu().recommend()
            print(recommendation)
        else:
            print('Введите y или n')

programm()
while True:
    again = input('Поищем еще? y/n: ')
    if again == 'y':
        programm()
    else:
        print('До свидания!')
        break
