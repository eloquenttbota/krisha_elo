"""Иллюстративные HTML-схемы того, как Decision Tree / Random Forest / XGBoost
принимают решение — с вопросами и признаками из этого же проекта.

Это не буквальный экспорт обученного дерева (там сотни узлов и его невозможно
показать на слайде) — это упрощённая, но честная по логике схема на 2 уровня,
чтобы нетехническая аудитория увидела сам механизм на понятном примере.
"""
from IPython.display import display, HTML

_TREE_CSS = """
<style>
.krisha-tree, .krisha-tree ul {
  padding-top: 20px; position: relative;
  display: flex; justify-content: center;
}
.krisha-tree li {
  list-style-type: none;
  position: relative;
  padding: 20px 8px 0 8px;
  text-align: center;
}
.krisha-tree li::before, .krisha-tree li::after {
  content: '';
  position: absolute; top: 0; right: 50%;
  border-top: 2px solid #CBD5E0;
  width: 50%; height: 20px;
}
.krisha-tree li::after {
  right: auto; left: 50%;
  border-left: 2px solid #CBD5E0;
}
.krisha-tree li:only-child::after, .krisha-tree li:only-child::before { display: none; }
.krisha-tree li:only-child { padding-top: 0; }
.krisha-tree li:first-child::before, .krisha-tree li:last-child::after { border: 0 none; }
.krisha-tree li:last-child::before { border-right: 2px solid #CBD5E0; border-radius: 0 5px 0 0; }
.krisha-tree li:first-child::after { border-radius: 5px 0 0 0; }
.krisha-tree ul ul::before {
  content: '';
  position: absolute; top: 0; left: 50%;
  border-left: 2px solid #CBD5E0;
  width: 0; height: 20px;
}
.krisha-node {
  display: inline-block;
  border-radius: 10px;
  padding: 10px 14px;
  font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  font-size: 12.5px;
  line-height: 1.35;
  min-width: 100px;
  box-sizing: border-box;
}
</style>
"""


def _node_html(label: str, colors: dict, leaf: bool = False) -> str:
    if leaf:
        return (f'<div class="krisha-node" style="background:{colors["pos"]};'
                f'color:white;font-weight:700;border:1px solid {colors["pos"]};">{label}</div>')
    return (f'<div class="krisha-node" style="background:white;color:#1A202C;'
            f'font-weight:600;border:1.5px solid {colors["primary"]};">{label}</div>')


def _render(node: dict, colors: dict) -> str:
    is_leaf = not node.get("children")
    html = f'<li>{_node_html(node["label"], colors, leaf=is_leaf)}'
    if not is_leaf:
        html += "<ul>"
        for child in node["children"]:
            html += _render(child, colors)
        html += "</ul>"
    html += "</li>"
    return html


def _wrap(title: str, subtitle: str, body_html: str, footnote: str, colors: dict) -> str:
    return f"""
    {_TREE_CSS}
    <div style="border:1px solid #E2E8F0;border-radius:12px;padding:18px 20px;margin:10px 0;
                font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:white;">
      <div style="font-weight:700;font-size:15px;color:#1A202C;">{title}</div>
      <div style="color:#718096;font-size:12px;margin-top:2px;margin-bottom:6px;">{subtitle}</div>
      {body_html}
      <div style="margin-top:14px;padding:10px 14px;background:#F7FAFC;border-radius:8px;
                  color:#4A5568;font-size:12.5px;">{footnote}</div>
    </div>
    """


# ─── Decision Tree: одно дерево, вопрос → вопрос → лист ────────────────────

def draw_decision_tree_html(colors: dict) -> None:
    tree = {
        "label": "Площадь ≥ 60 м²?",
        "children": [
            {
                "label": "Нет →<br>Район = Есильский?",
                "children": [
                    {"label": "Нет<br><b>≈ 480 000 тг/м²</b>"},
                    {"label": "Да<br><b>≈ 650 000 тг/м²</b>"},
                ],
            },
            {
                "label": "Да →<br>Год постройки ≥ 2015?",
                "children": [
                    {"label": "Нет<br><b>≈ 560 000 тг/м²</b>"},
                    {"label": "Да<br><b>≈ 740 000 тг/м²</b>"},
                ],
            },
        ],
    }
    body = f'<ul class="krisha-tree">{_render(tree, colors)}</ul>'
    html = _wrap(
        "🌲 Decision Tree — логика одного дерева на примере оценки квартиры",
        "Синие блоки — вопросы «да/нет», зелёные — итоговый лист с предсказанной ценой.",
        body,
        "Дерево физически не может предсказать ничего, кроме одного из своих листьев — "
        "поэтому у него ограниченный, конечный набор возможных ответов.",
        colors,
    )
    display(HTML(html))


# ─── Random Forest: несколько независимых деревьев + усреднение ────────────

def draw_random_forest_html(colors: dict) -> None:
    trees = [
        {
            "label": "Дерево А (случайная подвыборка 1)",
            "root": {
                "label": "Площадь ≥ 60 м²?",
                "children": [
                    {"label": "Нет<br><b>520 000</b>"},
                    {"label": "Да<br><b>700 000</b>"},
                ],
            },
            "pick": "700 000",
        },
        {
            "label": "Дерево Б (случайная подвыборка 2)",
            "root": {
                "label": "Район = Есильский?",
                "children": [
                    {"label": "Нет<br><b>560 000</b>"},
                    {"label": "Да<br><b>750 000</b>"},
                ],
            },
            "pick": "750 000",
        },
        {
            "label": "Дерево В (случайная подвыборка 3)",
            "root": {
                "label": "Высота потолков ≥ 3 м?",
                "children": [
                    {"label": "Нет<br><b>600 000</b>"},
                    {"label": "Да<br><b>780 000</b>"},
                ],
            },
            "pick": "780 000",
        },
    ]
    body = '<div style="display:flex;gap:18px;flex-wrap:wrap;justify-content:center;">'
    for t in trees:
        body += (
            f'<div style="text-align:center;">'
            f'<div style="font-size:11.5px;color:#718096;margin-bottom:2px;">{t["label"]}</div>'
            f'<ul class="krisha-tree" style="padding-top:10px;">{_render(t["root"], colors)}</ul>'
            f"</div>"
        )
    body += "</div>"
    html = _wrap(
        "🌳🌳🌳 Random Forest — много разных деревьев на разных случайных подвыборках",
        "Каждое дерево видит свою часть данных и признаков, поэтому задаёт разные вопросы.",
        body,
        "Пример: квартира 75 м², Есильский р-н, потолки 3.1 м → "
        "Дерево А даёт 700 000, Дерево Б — 750 000, Дерево В — 780 000. "
        "Финальный прогноз — среднее по всем деревьям леса: <b>≈ 743 000 тг/м²</b>. "
        "Это и есть «мудрость толпы» — отдельные деревья ошибаются по-разному, "
        "а их среднее оказывается устойчивее любого из них.",
        colors,
    )
    display(HTML(html))


# ─── XGBoost: цепочка деревьев, каждое исправляет ошибку предыдущего ───────

def draw_xgboost_html(colors: dict) -> None:
    trees = [
        {
            "title": "Дерево 1 — грубый первый прогноз",
            "root": {
                "label": "Площадь ≥ 60 м²?",
                "children": [
                    {"label": "Нет<br><b>550 000</b>"},
                    {"label": "Да<br><b>650 000</b>"},
                ],
            },
        },
        {
            "title": "Дерево 2 — коррекция для премиум-сегмента",
            "root": {
                "label": "Есильский р-н И<br>потолки ≥ 3 м?",
                "children": [
                    {"label": "Нет<br><b>+0</b>"},
                    {"label": "Да<br><b>+90 000</b>"},
                ],
            },
        },
        {
            "title": "Дерево 3 — точечная докрутка",
            "root": {
                "label": "Год постройки ≥ 2015?",
                "children": [
                    {"label": "Нет<br><b>−10 000</b>"},
                    {"label": "Да<br><b>+20 000</b>"},
                ],
            },
        },
    ]
    body = '<div style="display:flex;gap:8px;flex-wrap:wrap;justify-content:center;align-items:center;">'
    for i, t in enumerate(trees):
        body += (
            f'<div style="text-align:center;">'
            f'<div style="font-size:11.5px;color:#718096;margin-bottom:2px;max-width:150px;">{t["title"]}</div>'
            f'<ul class="krisha-tree" style="padding-top:10px;">{_render(t["root"], colors)}</ul>'
            f"</div>"
        )
        if i < len(trees) - 1:
            body += (f'<div style="font-size:22px;color:{colors["neg"]};font-weight:700;'
                      f'align-self:center;">→</div>')
    body += "</div>"
    html = _wrap(
        "🚀 XGBoost — цепочка деревьев, каждое исправляет ошибки предыдущего",
        "В отличие от Random Forest деревья строятся не параллельно, а по очереди.",
        body,
        "Пример: та же квартира 75 м², Есильский р-н, потолки 3.1 м, год 2021 → "
        "Дерево 1 даёт грубую базу 650 000, Дерево 2 добавляет поправку +90 000 за "
        "премиум-сегмент, Дерево 3 добавляет ещё +20 000 за новый год постройки. "
        "Итог — сумма всей цепочки: <b>650 000 + 90 000 + 20 000 = 760 000 тг/м²</b>.",
        colors,
    )
    display(HTML(html))
