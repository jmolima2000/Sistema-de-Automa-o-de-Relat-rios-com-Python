from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class ReportConfig:
    input_file: Path
    output_dir: Path = Path("saida_relatorios")
    date_column: str = "data"
    category_column: str = "categoria"
    amount_column: str = "valor"
    status_column: Optional[str] = "status"
    customer_column: Optional[str] = "cliente"
    decimal_places: int = 2


class ReportAutomationSystem:
    def __init__(self, config: ReportConfig):
        self.config = config
        self.df: Optional[pd.DataFrame] = None
        self.kpis: dict[str, object] = {}
        self.output_dir = self.config.output_dir
        self.charts_dir = self.output_dir / "graficos"
        self.tables_dir = self.output_dir / "tabelas"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        print("[1/7] Lendo arquivo...")
        self.load_data()

        print("[2/7] Limpando dados...")
        self.clean_data()

        print("[3/7] Calculando KPIs...")
        self.calculate_kpis()

        print("[4/7] Gerando tabelas analíticas...")
        summary_tables = self.generate_summary_tables()

        print("[5/7] Gerando gráficos...")
        charts = self.generate_charts(summary_tables)

        print("[6/7] Exportando relatórios...")
        self.export_excel(summary_tables)
        self.export_html(summary_tables, charts)

        print("[7/7] Processo finalizado com sucesso.")
        print(f"Relatórios salvos em: {self.output_dir.resolve()}")

    def load_data(self) -> None:
        file_path = self.config.input_file
        if not file_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            self.df = pd.read_csv(file_path)
        elif suffix in {".xlsx", ".xls"}:
            self.df = pd.read_excel(file_path)
        else:
            raise ValueError("Formato não suportado. Use CSV, XLSX ou XLS.")

        if self.df is None or self.df.empty:
            raise ValueError("O arquivo foi lido, mas não contém dados.")

    def clean_data(self) -> None:
        assert self.df is not None
        df = self.df.copy()

        df.columns = [col.strip().lower() for col in df.columns]

        required_columns = {
            self.config.date_column,
            self.config.category_column,
            self.config.amount_column,
        }

        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(
                "As seguintes colunas obrigatórias não existem no arquivo: "
                + ", ".join(missing)
            )

        df[self.config.date_column] = pd.to_datetime(
            df[self.config.date_column], errors="coerce"
        )
        df[self.config.amount_column] = pd.to_numeric(
            df[self.config.amount_column], errors="coerce"
        )

        df[self.config.category_column] = (
            df[self.config.category_column]
            .astype(str)
            .str.strip()
            .replace({"": "Não informado"})
        )

        if self.config.status_column and self.config.status_column in df.columns:
            df[self.config.status_column] = (
                df[self.config.status_column]
                .astype(str)
                .str.strip()
                .replace({"": "Não informado"})
            )

        if self.config.customer_column and self.config.customer_column in df.columns:
            df[self.config.customer_column] = (
                df[self.config.customer_column]
                .astype(str)
                .str.strip()
                .replace({"": "Não informado"})
            )

        df = df.dropna(subset=[self.config.date_column, self.config.amount_column])
        df = df.drop_duplicates().sort_values(by=self.config.date_column)

        df["ano"] = df[self.config.date_column].dt.year
        df["mes"] = df[self.config.date_column].dt.month
        df["mes_ano"] = df[self.config.date_column].dt.to_period("M").astype(str)
        df["dia"] = df[self.config.date_column].dt.day

        self.df = df.reset_index(drop=True)

    def calculate_kpis(self) -> None:
        assert self.df is not None
        df = self.df
        amount = self.config.amount_column
        status_col = self.config.status_column

        total_registros = len(df)
        valor_total = round(float(df[amount].sum()), self.config.decimal_places)
        ticket_medio = round(float(df[amount].mean()), self.config.decimal_places)
        maior_valor = round(float(df[amount].max()), self.config.decimal_places)
        menor_valor = round(float(df[amount].min()), self.config.decimal_places)
        categorias_unicas = int(df[self.config.category_column].nunique())
        periodo_inicial = df[self.config.date_column].min().strftime("%d/%m/%Y")
        periodo_final = df[self.config.date_column].max().strftime("%d/%m/%Y")

        kpis = {
            "Total de Registros": total_registros,
            "Valor Total": valor_total,
            "Ticket Médio": ticket_medio,
            "Maior Valor": maior_valor,
            "Menor Valor": menor_valor,
            "Categorias Únicas": categorias_unicas,
            "Período Inicial": periodo_inicial,
            "Período Final": periodo_final,
        }

        if status_col and status_col in df.columns:
            concluidos = int(
                df[status_col].astype(str).str.lower().isin(["concluído", "concluido", "pago", "finalizado"]).sum()
            )
            taxa = round((concluidos / total_registros) * 100, 2) if total_registros else 0
            kpis["Registros Concluídos"] = concluidos
            kpis["Taxa de Conclusão (%)"] = taxa

        self.kpis = kpis

    def generate_summary_tables(self) -> dict[str, pd.DataFrame]:
        assert self.df is not None
        df = self.df
        amount = self.config.amount_column
        category = self.config.category_column

        resumo_mensal = (
            df.groupby("mes_ano", as_index=False)[amount]
            .agg(["sum", "mean", "count"])
            .reset_index()
            .rename(columns={"sum": "valor_total", "mean": "ticket_medio", "count": "quantidade"})
        )

        resumo_categoria = (
            df.groupby(category, as_index=False)[amount]
            .agg(["sum", "mean", "count"])
            .reset_index()
            .rename(columns={category: "categoria", "sum": "valor_total", "mean": "ticket_medio", "count": "quantidade"})
            .sort_values(by="valor_total", ascending=False)
        )

        resumo_ano = (
            df.groupby("ano", as_index=False)[amount]
            .agg(["sum", "mean", "count"])
            .reset_index()
            .rename(columns={"sum": "valor_total", "mean": "ticket_medio", "count": "quantidade"})
        )

        top_registros = df.sort_values(by=amount, ascending=False).head(10).copy()

        tables = {
            "base_tratada": df,
            "resumo_mensal": resumo_mensal,
            "resumo_categoria": resumo_categoria,
            "resumo_ano": resumo_ano,
            "top_10_registros": top_registros,
        }

        if self.config.status_column and self.config.status_column in df.columns:
            resumo_status = (
                df.groupby(self.config.status_column, as_index=False)[amount]
                .agg(["sum", "count"])
                .reset_index()
                .rename(columns={self.config.status_column: "status", "sum": "valor_total", "count": "quantidade"})
                .sort_values(by="valor_total", ascending=False)
            )
            tables["resumo_status"] = resumo_status

        if self.config.customer_column and self.config.customer_column in df.columns:
            resumo_cliente = (
                df.groupby(self.config.customer_column, as_index=False)[amount]
                .agg(["sum", "count"])
                .reset_index()
                .rename(columns={self.config.customer_column: "cliente", "sum": "valor_total", "count": "quantidade"})
                .sort_values(by="valor_total", ascending=False)
                .head(10)
            )
            tables["top_10_clientes"] = resumo_cliente

        for name, table in tables.items():
            table.to_csv(self.tables_dir / f"{name}.csv", index=False)

        return tables

    def generate_charts(self, tables: dict[str, pd.DataFrame]) -> dict[str, str]:
        charts: dict[str, str] = {}

        monthly = tables["resumo_mensal"]
        monthly_path = self.charts_dir / "evolucao_mensal.png"
        plt.figure(figsize=(10, 5))
        plt.plot(monthly["mes_ano"], monthly["valor_total"], marker="o")
        plt.title("Evolução Mensal do Valor Total")
        plt.xlabel("Mês")
        plt.ylabel("Valor Total")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(monthly_path)
        plt.close()
        charts["Evolução Mensal"] = monthly_path.name

        category = tables["resumo_categoria"].head(10)
        category_path = self.charts_dir / "top_categorias.png"
        plt.figure(figsize=(10, 6))
        plt.bar(category["categoria"], category["valor_total"])
        plt.title("Top Categorias por Valor Total")
        plt.xlabel("Categoria")
        plt.ylabel("Valor Total")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(category_path)
        plt.close()
        charts["Top Categorias"] = category_path.name

        if "resumo_status" in tables:
            status = tables["resumo_status"]
            status_path = self.charts_dir / "status_distribuicao.png"
            plt.figure(figsize=(8, 8))
            plt.pie(status["valor_total"], labels=status["status"], autopct="%1.1f%%")
            plt.title("Distribuição por Status")
            plt.tight_layout()
            plt.savefig(status_path)
            plt.close()
            charts["Distribuição por Status"] = status_path.name

        return charts

    def export_excel(self, tables: dict[str, pd.DataFrame]) -> None:
        excel_path = self.output_dir / "relatorio_geral.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            pd.DataFrame(list(self.kpis.items()), columns=["Indicador", "Valor"]).to_excel(
                writer, index=False, sheet_name="KPIs"
            )
            for sheet_name, table in tables.items():
                safe_sheet_name = sheet_name[:31]
                table.to_excel(writer, index=False, sheet_name=safe_sheet_name)

    def export_html(self, tables: dict[str, pd.DataFrame], charts: dict[str, str]) -> None:
        html_path = self.output_dir / "relatorio_geral.html"
        kpi_html = "".join(
            f'<div class="kpi-card"><h3>{key}</h3><p>{value}</p></div>'
            for key, value in self.kpis.items()
        )

        tables_html = ""
        for name, table in tables.items():
            title = name.replace("_", " ").title()
            tables_html += f"<h2>{title}</h2>"
            tables_html += table.head(20).to_html(index=False, classes="table", border=0)

        charts_html = ""
        for title, file_name in charts.items():
            charts_html += f'<div class="chart-box"><h2>{title}</h2><img src="graficos/{file_name}" alt="{title}"></div>'

        html_content = f"""
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório Automatizado</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 30px; background: #f4f6f9; color: #333; }}
        h1 {{ color: #1f3c88; }}
        h2 {{ margin-top: 35px; color: #2d3436; }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin: 20px 0 30px; }}
        .kpi-card {{ background: #ffffff; padding: 18px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
        .kpi-card h3 {{ margin: 0 0 8px; font-size: 15px; color: #1f3c88; }}
        .kpi-card p {{ margin: 0; font-size: 20px; font-weight: bold; }}
        .table {{ width: 100%; border-collapse: collapse; background: #fff; }}
        .table th, .table td {{ border: 1px solid #ddd; padding: 10px; text-align: left; font-size: 14px; }}
        .table th {{ background: #1f3c88; color: #fff; }}
        .chart-box {{ margin: 30px 0; background: #fff; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
        .chart-box img {{ width: 100%; max-width: 900px; display: block; margin: 10px auto; }}
    </style>
</head>
<body>
    <h1>Relatório Automatizado</h1>
    <p>Arquivo de origem: {self.config.input_file.name}</p>
    <p>Período analisado: {self.kpis.get('Período Inicial', '-')} até {self.kpis.get('Período Final', '-')}</p>

    <div class="kpi-grid">{kpi_html}</div>
    {charts_html}
    {tables_html}
</body>
</html>
        """

        html_path.write_text(html_content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sistema de automação de relatórios com Python, Pandas e Matplotlib."
    )
    parser.add_argument("input_file", help="Caminho do arquivo CSV ou Excel.")
    parser.add_argument(
        "--output-dir",
        default="saida_relatorios",
        help="Diretório onde os relatórios serão salvos.",
    )
    parser.add_argument("--date-column", default="data", help="Nome da coluna de data.")
    parser.add_argument(
        "--category-column", default="categoria", help="Nome da coluna de categoria."
    )
    parser.add_argument("--amount-column", default="valor", help="Nome da coluna de valor.")
    parser.add_argument("--status-column", default="status", help="Nome da coluna de status.")
    parser.add_argument("--customer-column", default="cliente", help="Nome da coluna de cliente.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = ReportConfig(
        input_file=Path(args.input_file),
        output_dir=Path(args.output_dir),
        date_column=args.date_column.lower(),
        category_column=args.category_column.lower(),
        amount_column=args.amount_column.lower(),
        status_column=args.status_column.lower() if args.status_column else None,
        customer_column=args.customer_column.lower() if args.customer_column else None,
    )

    system = ReportAutomationSystem(config)
    system.run()


if __name__ == "__main__":
    main()
