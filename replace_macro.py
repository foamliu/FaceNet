# -*- coding: utf-8 -*-
import json
if __name__ == '__main__':
    with open('result.json', 'r', encoding="utf-8") as file:
        result = json.load(file)

    with open('README.t', 'r', encoding="utf-8") as file:
        text = file.readlines()

        text = ''.join(text)

    for i in range(10):
        key_distance_a_p = 'distance_{}_a_p'.format(i)
        template = text.replace('$({})', )

    for i in range(0, 10):
        beam_data = [line.strip() for line in beam[i * 4:(i + 1) * 4]]
        beam_text = '<br>'.join(beam_data)
        template = template.replace('({})'.format(i), beam_text)

    with open('README.md', 'w', encoding="utf-8") as file:
        file.write(template)
