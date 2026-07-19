"""Общие визуальные помощники для презентационного ноутбука.

Единственная задача этих функций — рендерить аккуратные HTML-карточки
с результатами, чтобы на экране во время презентации не было кода,
только выводы.
"""
import base64
import os

from IPython.display import display, HTML


def _logo_img_tag(logo_path: str, size: int = 28) -> str:
    if not logo_path or not os.path.exists(logo_path):
        return ""
    with open(logo_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return (
        f'<img src="data:image/png;base64,{b64}" width="{size}" height="{size}" '
        f'style="border-radius:6px;object-fit:cover;flex-shrink:0;" />'
    )


def card(title: str, rows: list[tuple[str, str]] | None = None,
         subtitle: str | None = None, accent: str = "#2F80ED",
         note: str | None = None, logo_path: str | None = None) -> None:
    """Рисует карточку-результат: заголовок + таблица key/value + заметка."""
    rows_html = ""
    if rows:
        for label, value in rows:
            rows_html += (
                f'<div style="display:flex;justify-content:space-between;'
                f'padding:6px 0;border-bottom:1px solid #EDF2F7;">'
                f'<span style="color:#718096;font-size:13px;">{label}</span>'
                f'<span style="color:#1A202C;font-weight:600;font-size:13px;">{value}</span>'
                f'</div>'
            )
    subtitle_html = (
        f'<div style="color:#718096;font-size:12px;margin-top:2px;">{subtitle}</div>'
        if subtitle else ""
    )
    note_html = (
        f'<div style="margin-top:10px;padding:8px 12px;background:#F7FAFC;'
        f'border-radius:8px;color:#4A5568;font-size:12px;">{note}</div>'
        if note else ""
    )
    logo_html = _logo_img_tag(logo_path) if logo_path else ""
    title_row = (
        f'<div style="display:flex;align-items:center;gap:8px;">'
        f'{logo_html}<span style="font-weight:700;font-size:15px;color:#1A202C;">{title}</span>'
        f'</div>'
        if logo_html else
        f'<div style="font-weight:700;font-size:15px;color:#1A202C;">{title}</div>'
    )
    html = f"""
    <div style="border:1px solid #E2E8F0;border-left:4px solid {accent};
                border-radius:10px;padding:14px 18px;margin:10px 0;
                font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
                background:white;">
      {title_row}
      {subtitle_html}
      <div style="margin-top:8px;">{rows_html}</div>
      {note_html}
    </div>
    """
    display(HTML(html))


def hypothesis_result(tag: str, text: str, stat_name: str, stat_value: float,
                       p_value: float, alpha: float = 0.05) -> bool:
    """Печатает результат проверки гипотезы в единообразном, понятном виде."""
    confirmed = p_value < alpha
    icon = "✅ подтверждена" if confirmed else "❌ не подтверждена"
    color = "#27AE60" if confirmed else "#EB5757"
    html = f"""
    <div style="display:flex;align-items:center;gap:12px;padding:10px 14px;
                border-radius:8px;background:#F7FAFC;margin:4px 0;
                font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
      <span style="font-weight:700;color:{color};min-width:34px;">{tag}</span>
      <span style="flex:1;color:#2D3748;font-size:13px;">{text}</span>
      <span style="color:#718096;font-size:12px;">{stat_name}={stat_value:.3f}, p={p_value:.4f}</span>
      <span style="color:{color};font-weight:600;font-size:12px;white-space:nowrap;">{icon}</span>
    </div>
    """
    display(HTML(html))
    return confirmed


def show_qr_code(path: str, caption: str, size: int = 300) -> None:
    """Показывает QR-код с подписью, если файл уже лежит в директории проекта.
    Пока файла нет — аккуратная заглушка с именем, которое нужно положить рядом."""
    if not os.path.exists(path):
        html = f"""
        <div style="display:inline-block;width:{size}px;height:{size}px;margin:8px 16px 8px 0;
                    border:2px dashed #CBD5E0;border-radius:10px;text-align:center;
                    display:flex;flex-direction:column;align-items:center;justify-content:center;
                    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
                    color:#A0AEC0;font-size:12px;padding:8px;box-sizing:border-box;">
          <span style="font-size:22px;">🔲</span>
          <span>Положите файл<br><b>{path}</b><br>рядом с ноутбуком</span>
        </div>
        """
        display(HTML(html))
        return

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    html = f"""
    <div style="display:inline-block;text-align:center;margin:8px 16px 8px 0;
                font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
      <img src="data:image/png;base64,{b64}" width="{size}" height="{size}"
           style="border-radius:10px;border:1px solid #E2E8F0;" />
      <div style="margin-top:6px;font-size:12px;color:#4A5568;max-width:{size}px;">{caption}</div>
    </div>
    """
    display(HTML(html))


def section_banner(number: str, title: str, description: str, icon: str = "📊",
                    color: str = "#2F80ED") -> None:
    """Крупный баннер начала раздела — используется как замена markdown-заголовков."""
    html = f"""
    <div style="display:flex;align-items:center;gap:14px;
                background:linear-gradient(135deg,{color}18,{color}05);
                border-left:4px solid {color};border-radius:10px;
                padding:16px 20px;margin:18px 0 10px 0;
                font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
      <span style="font-size:26px;">{icon}</span>
      <div>
        <div style="font-size:11px;color:{color};font-weight:700;letter-spacing:1px;">{number}</div>
        <div style="font-size:17px;font-weight:700;color:#1A202C;">{title}</div>
        <div style="font-size:12px;color:#718096;margin-top:2px;">{description}</div>
      </div>
    </div>
    """
    display(HTML(html))
