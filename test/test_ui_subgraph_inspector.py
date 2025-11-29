import re


def test_embedded_subgraph_handlers_exist():
    with open('static/designer.js', 'r', encoding='utf-8') as f:
        src = f.read()

    assert 'EmbeddedSubgraphNode.prototype.onSelected' in src, "EmbeddedSubgraphNode.onSelected not found"
    assert 'EmbeddedSubgraphNode.prototype.onDeselected' in src, "EmbeddedSubgraphNode.onDeselected not found"
    assert 'SubgraphNode.prototype.onSelected' in src, "SubgraphNode.onSelected not found"
    assert 'SubgraphRefNode.prototype.onSelected' in src, "SubgraphRefNode.onSelected not found"


def test_embedded_title_sets_template_name():
    with open('static/designer.js', 'r', encoding='utf-8') as f:
        src = f.read()

    # Ensure title input change event uses setProperty for embedded subgraph
    pattern = re.compile(r"titleInput.addEventListener\([\s\S]*?setProperty\s*\(\s*['\"]template_name['\"]", re.MULTILINE)
    assert pattern.search(src), 'title input change does not set template_name for embedded subgraphs'
