# -*- coding: utf-8 -*-
import json

def replace():
    with open('result.json', 'r', encoding="utf-8") as file:
        result = json.load(file)

    with open('README.t', 'r', encoding="utf-8") as file:
        text = file.readlines()

        text = ''.join(text)

    for i in range(10):
        distance_a_p = 'distance_{}_a_p'.format(i)
        text = text.replace('$({})'.format(distance_a_p), "{0:.2f}".format(result[distance_a_p]))
        distance_a_n = 'distance_{}_a_n'.format(i)
        text = text.replace('$({})'.format(distance_a_n), "{0:.2f}".format(result[distance_a_n]))

    with open('README.md', 'w', encoding="utf-8") as file:
        file.write(text)

if __name__ == '__main__':
    replace()
