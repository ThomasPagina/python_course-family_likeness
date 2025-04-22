#!/usr/bin/env python3

import requests
from bs4 import BeautifulSoup
from graphviz import Digraph

def fetch_page(lang, title):
    url = f'https://{lang}.wikipedia.org/wiki/{title}'
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def parse_structure(html):
    soup = BeautifulSoup(html, 'html.parser')
    content = soup.find(id='mw-content-text')
    tree = []
    stack = []

    for tag in content.find_all(['h2','h3','h4','h5','h6','figure']):
        if tag.name.startswith('h'):
            level = int(tag.name[1])
            text = tag.get_text(strip=True)
            while stack and stack[-1][0] >= level:
                stack.pop()
            node = {'title': text, 'children': [], 'level': level}
            if stack:
                stack[-1][2]['children'].append(node)
            else:
                tree.append(node)
            stack.append((level, text, node))
        elif tag.name == 'figure':
            caption = tag.find('figcaption').get_text(strip=True) if tag.find('figcaption') else ''
            img_node = {'title': f'Images:\n{caption}', 'children': [], 'level': None, 'is_image_list': True}
            if stack:
                stack[-1][2]['children'].append(img_node)
            else:
                tree.append(img_node)

    return tree

def build_graph(tree, graph=None, parent=None):
    if graph is None:
        graph = Digraph(comment='Wikipedia Structure', engine='neato')
        graph.graph_attr.update(
            overlap='false', sep='1', splines='true', rankdir='TB', dpi='300', nodesep='1', ranksep='1'
        )

    for node in tree:
        nid = str(id(node))
        attrs = {}
        title = node['title']

        if node.get('is_image_list'):
            attrs['shape'] = 'box'
            attrs['style'] = 'solid'
            attrs['fillcolor'] = 'white'
            attrs['label'] = title
        else:
            attrs['shape'] = 'ellipse'
            attrs['style'] = 'filled'
            attrs['fillcolor'] = 'lightgrey'
            lvl = node['level'] or 3
           
            if lvl in (1, 2):
                attrs['label'] = f'<<B>{title}</B>>'
            elif lvl == 3:
                attrs['label'] = title
            else:
                attrs['label'] = f'<<I>{title}</I>>'
                
        graph.node(nid, **attrs)
        if parent:
            graph.edge(parent, nid)
        if node['children']:
            build_graph(node['children'], graph, nid)

    return graph

def generate_graph(lang, title):
    html = fetch_page(lang, title)
    tree = parse_structure(html)
    return build_graph(tree)

def main():
    for lang, title in [('de', 'Rembrandt_van_Rijn'), ('en', 'Rembrandt')]:
        graph = generate_graph(lang, title)
        graph.format = 'png'
        out = graph.render(f'{lang}_{title}', view=True)
        print('Gerenderte Grafik:', out)

if __name__ == '__main__':
    main()
