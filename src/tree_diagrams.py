"""Иллюстративные HTML-схемы того, как Decision Tree / Random Forest / XGBoost
принимают решение — с вопросами и признаками из этого же проекта.

Это не буквальный экспорт обученного дерева (там сотни узлов, на слайде не
показать) — упрощённая, но честная по логике схема на 2 уровня, чтобы
нетехническая аудитория увидела сам механизм на понятном примере.

Ветвление рисуется вложенными блоками с цветной рамкой слева (а не
CSS-коннекторами через :only-child/::before — они не одинаково стабильно
рендерятся в разных HTML-движках, включая встроенный просмотрщик ноутбуков),
поэтому схема выглядит одинаково независимо от того, где открыт ноутбук.
"""
from IPython.display import display, HTML

_FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif"


def _question_html(label: str, colors: dict) -> str:
    return (
        f'<div style="display:inline-block;background:linear-gradient(135deg,{colors["primary"]},#2563C9);'
        f'color:white;font-weight:700;font-size:13px;padding:10px 18px;border-radius:999px;'
        f'box-shadow:0 3px 8px rgba(47,128,237,0.35);font-family:{_FONT};">❓ {label}</div>'
    )


def _leaf_html(label: str, colors: dict) -> str:
    return (
        f'<div style="display:inline-block;background:linear-gradient(135deg,{colors["pos"]},#1E8449);'
        f'color:white;font-weight:700;font-size:13px;padding:10px 18px;border-radius:10px;'
        f'box-shadow:0 3px 8px rgba(39,174,96,0.35);font-family:{_FONT};">💰 {label}</div>'
    )


def _branch_html(answer: str, node_html: str, good: bool, colors: dict) -> str:
    """Содержимое одной ветки «Да»/«Нет» — цветная рамка слева группирует
    вопрос/лист без хрупких линий-коннекторов."""
    tag_color = colors["pos"] if good else colors["neg"]
    tag_text = answer or ("✅ Да" if good else "❌ Нет")
    return f"""
    <div style="border-left:3px solid {tag_color};border-radius:0 10px 10px 0;
                background:{tag_color}0D;padding:12px 16px;height:100%;box-sizing:border-box;">
      <div style="font-size:11px;font-weight:700;color:{tag_color};margin-bottom:8px;
                  text-transform:uppercase;letter-spacing:0.5px;">{tag_text}</div>
      {node_html}
    </div>
    """


def _render(node: dict, colors: dict, horizontal: bool = False) -> str:
    """horizontal=True кладёт две ветки рядом (колонками) — годится для
    самостоятельного дерева. horizontal=False укладывает их друг под другом —
    компактнее для узких колонок мини-деревьев (лес/бустинг)."""
    is_leaf = "children" not in node
    if is_leaf:
        return _leaf_html(node["label"], colors)

    html = _question_html(node["label"], colors)
    branches = [
        _branch_html(c.get("answer", ""), _render(c["node"], colors, horizontal), c["good"], colors)
        for c in node["children"]
    ]
    if horizontal:
        cols = "".join(f'<div style="flex:1;min-width:0;">{b}</div>' for b in branches)
        html += f'<div style="display:flex;gap:14px;margin-top:12px;align-items:stretch;">{cols}</div>'
    else:
        stacked = "".join(f"<div>{b}</div>" for b in branches)
        html += f'<div style="display:flex;flex-direction:column;gap:10px;margin-top:10px;">{stacked}</div>'
    return html


def _card(title: str, subtitle: str, body_html: str, footnote: str, accent: str) -> str:
    return f"""
    <div style="border:1px solid #E2E8F0;border-radius:14px;padding:20px 22px;margin:12px 0;
                font-family:{_FONT};background:linear-gradient(180deg,#FFFFFF,#FAFBFF);
                box-shadow:0 2px 10px rgba(0,0,0,0.04);">
      <div style="font-weight:700;font-size:16px;color:#1A202C;">{title}</div>
      <div style="color:#718096;font-size:12.5px;margin-top:2px;margin-bottom:14px;">{subtitle}</div>
      {body_html}
      <div style="margin-top:16px;padding:12px 16px;background:#F7FAFC;border-radius:10px;
                  color:#4A5568;font-size:12.5px;border-left:3px solid {accent};">{footnote}</div>
    </div>
    """


# ─── Decision Tree: одно дерево, вопрос → вопрос → лист ────────────────────

def draw_decision_tree_html(colors: dict) -> None:
    tree = {
        "label": "Площадь ≥ 60 м²?",
        "children": [
            {"good": False, "node": {
                "label": "Район = Есильский?",
                "children": [
                    {"good": False, "node": {"label": "≈ 480 000 тг/м²"}},
                    {"good": True, "node": {"label": "≈ 650 000 тг/м²"}},
                ],
            }},
            {"good": True, "node": {
                "label": "Год постройки ≥ 2015?",
                "children": [
                    {"good": False, "node": {"label": "≈ 560 000 тг/м²"}},
                    {"good": True, "node": {"label": "≈ 740 000 тг/м²"}},
                ],
            }},
        ],
    }
    body = (f'<div style="max-width:820px;margin:0 auto;text-align:center;">'
            f'{_render(tree, colors, horizontal=True)}</div>')
    html = _card(
        "🌲 Как Decision Tree оценивает квартиру",
        "Синий пузырь — вопрос «да/нет», зелёный — итоговый лист с предсказанной ценой.",
        body,
        "Дерево физически не может предсказать ничего, кроме одного из своих листьев — "
        "у него ограниченный, конечный набор возможных ответов.",
        colors["primary"],
    )
    display(HTML(html))


# ─── Random Forest: несколько независимых деревьев + усреднение ────────────

def draw_random_forest_html(colors: dict) -> None:
    trees = [
        ("Дерево А", "случайная подвыборка 1", "Площадь ≥ 60 м²?", "520 000", "700 000"),
        ("Дерево Б", "случайная подвыборка 2", "Район = Есильский?", "560 000", "750 000"),
        ("Дерево В", "случайная подвыборка 3", "Высота потолков ≥ 3 м?", "600 000", "780 000"),
    ]
    cols = ""
    for name, subtitle, q, no_val, yes_val in trees:
        mini = {
            "label": q,
            "children": [
                {"good": False, "node": {"label": no_val}},
                {"good": True, "node": {"label": yes_val}},
            ],
        }
        cols += f"""
        <div style="flex:1;min-width:220px;text-align:center;">
          <div style="font-weight:700;font-size:12.5px;color:#1A202C;">{name}</div>
          <div style="font-size:11px;color:#A0AEC0;margin-bottom:8px;">{subtitle}</div>
          {_render(mini, colors)}
        </div>
        """
    body = f'<div style="display:flex;gap:14px;flex-wrap:wrap;justify-content:center;">{cols}</div>'
    html = _card(
        "🌳🌳🌳 Как Random Forest оценивает квартиру",
        "Каждое дерево видит свою случайную часть данных и признаков — поэтому задаёт разные вопросы.",
        body,
        "Пример: квартира 75 м², Есильский р-н, потолки 3.1 м → Дерево А даёт 700 000, "
        "Дерево Б — 750 000, Дерево В — 780 000. Финальный прогноз — среднее по всем "
        "деревьям леса: <b>≈ 743 000 тг/м²</b>. Это и есть «мудрость толпы»: отдельные "
        "деревья ошибаются по-разному, а их среднее оказывается устойчивее любого из них.",
        colors["pos"],
    )
    display(HTML(html))


# ─── XGBoost: цепочка деревьев, каждое исправляет ошибку предыдущего ───────

def draw_xgboost_html(colors: dict) -> None:
    steps = [
        ("Дерево 1", "грубый первый прогноз", "Площадь ≥ 60 м²?", "550 000", "650 000"),
        ("Дерево 2", "коррекция для премиум-сегмента", "Есильский р-н И потолки ≥ 3 м?", "+0", "+90 000"),
        ("Дерево 3", "точечная докрутка", "Год постройки ≥ 2015?", "−10 000", "+20 000"),
    ]
    cols = []
    for name, subtitle, q, no_val, yes_val in steps:
        mini = {
            "label": q,
            "children": [
                {"good": False, "node": {"label": no_val}},
                {"good": True, "node": {"label": yes_val}},
            ],
        }
        cols.append(f"""
        <div style="text-align:center;min-width:220px;">
          <div style="font-weight:700;font-size:12.5px;color:#1A202C;">{name}</div>
          <div style="font-size:11px;color:#A0AEC0;margin-bottom:8px;max-width:200px;
                      margin-left:auto;margin-right:auto;">{subtitle}</div>
          {_render(mini, colors)}
        </div>
        """)
    arrow = (f'<div style="font-size:26px;color:{colors["accent"]};font-weight:700;'
             f'align-self:center;padding:0 4px;">➜</div>')
    body = (f'<div style="display:flex;gap:6px;flex-wrap:wrap;justify-content:center;'
            f'align-items:center;">{arrow.join(cols)}</div>')
    html = _card(
        "🚀 Как XGBoost оценивает квартиру",
        "Деревья строятся не параллельно, как в Random Forest, а по очереди — каждое следующее "
        "исправляет ошибки предыдущего.",
        body,
        "Пример: та же квартира 75 м², Есильский р-н, потолки 3.1 м, год 2021 → Дерево 1 "
        "даёт грубую базу 650 000, Дерево 2 добавляет поправку +90 000 за премиум-сегмент, "
        "Дерево 3 добавляет ещё +20 000 за новый год постройки. Итог — сумма всей цепочки: "
        "<b>650 000 + 90 000 + 20 000 = 760 000 тг/м²</b>.",
        colors["accent"],
    )
    display(HTML(html))
