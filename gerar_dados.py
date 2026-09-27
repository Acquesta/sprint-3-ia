"""Gera a base sintética de clientes do pós-venda (python gerar_dados.py).

Os números daqui são premissas nossas, sem base em dados da Ford. Cada linha é um veículo
na data de referência (30/06/2025), e `evadiu` = 1 quando o cliente não faz nenhum serviço
na rede nos 12 meses seguintes. A evasão sai de um logit com efeitos não lineares,
interações, uma "lealdade" que o CRM não vê e ruído, calibrado para dar ~29%. Depois a base
ganha a sujeira de um CRM de verdade (ausentes, texto fora do padrão, valores impossíveis,
outliers, duplicatas) e uma coluna que vaza o alvo.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "data" / "clientes_pos_venda.csv"
SEED = 42
N_CLIENTES = 10_000
DATA_REFERENCIA = "2025-06-30"
ANO_REFERENCIA = 2025
TARGET = "evadiu"
REGIOES = ["Sudeste", "Sul", "Nordeste", "Centro-Oeste", "Norte"]
MODELOS_FORA_DE_LINHA = ["Ka", "EcoSport"]

COLUNAS_CSV = [
    "id_cliente", "vin", "data_referencia", "modelo", "ano_modelo", "km_atual", "tipo_uso",
    "regiao", "distancia_concessionaria_km", "garantia_estendida", "plano_manutencao",
    "revisoes_rede_24m", "dias_desde_ultima_visita", "ticket_medio_rs", "nps_ultima_visita",
    "reclamacoes_12m", "recall_pendente", "sessoes_app_30d", "aceita_whatsapp",
    "idade_cliente", "motivo_saida_pesquisa", TARGET,
]

TAXA_EVASAO_ALVO = 0.29

# modelo: (participação na base, primeiro ano-modelo, último ano-modelo, ticket mediano de revisão em R$)
PERFIL_MODELOS = {
    "Ranger": (0.20, 2013, 2025, 1200),
    "Territory": (0.14, 2020, 2025, 950),
    "Bronco Sport": (0.10, 2021, 2025, 950),
    "Maverick": (0.08, 2022, 2025, 1000),
    "F-150": (0.04, 2023, 2025, 1800),
    "Mustang Mach-E": (0.03, 2022, 2025, 1100),
    "Transit": (0.06, 2021, 2025, 1300),
    "Ka": (0.20, 2012, 2021, 650),
    "EcoSport": (0.15, 2012, 2021, 750),
}
PARTICIPACAO_REGIOES = [0.45, 0.17, 0.20, 0.10, 0.08]
DISTANCIA_MEDIANA_KM = {"Sudeste": 8, "Sul": 12, "Nordeste": 18, "Centro-Oeste": 20, "Norte": 35}
KM_ANUAL_MEDIANO = {"particular": 12_000, "aplicativo": 45_000, "frota": 30_000}
MOTIVOS_SAIDA = ["preço da revisão", "oficina independente mais perto", "atendimento ruim", "vendeu o carro", "outro"]
CARACTERES_VIN = list("ABCDEFGHJKLMNPRSTUVWXYZ0123456789")  # VIN não usa I, O e Q

COLUNAS_INTEIRAS = [
    "ano_modelo", "km_atual", "garantia_estendida", "plano_manutencao", "revisoes_rede_24m",
    "dias_desde_ultima_visita", "nps_ultima_visita", "reclamacoes_12m", "recall_pendente",
    "sessoes_app_30d", "aceita_whatsapp", "idade_cliente", TARGET,
]


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _calibrar_intercepto(logit, taxa_alvo):
    """Acha, por busca binária, o intercepto que deixa a evasão média em `taxa_alvo`."""
    baixo, alto = -15.0, 15.0
    for _ in range(80):
        meio = (baixo + alto) / 2
        if _sigmoid(logit + meio).mean() < taxa_alvo:
            baixo = meio
        else:
            alto = meio
    return (baixo + alto) / 2


def _gerar_clientes(n, rng):
    """Base limpa, com os valores verdadeiros e o alvo."""
    nomes = list(PERFIL_MODELOS)
    participacao = np.array([PERFIL_MODELOS[m][0] for m in nomes])
    modelo = rng.choice(nomes, size=n, p=participacao / participacao.sum())
    primeiro_ano = np.array([PERFIL_MODELOS[m][1] for m in modelo])
    ultimo_ano = np.array([PERFIL_MODELOS[m][2] for m in modelo])
    ano_modelo = rng.integers(primeiro_ano, ultimo_ano + 1)
    idade = ANO_REFERENCIA - ano_modelo
    fora_de_linha = np.isin(modelo, MODELOS_FORA_DE_LINHA)

    pesados = np.isin(modelo, ["Ranger", "Transit", "F-150"])
    probs_uso = np.where(
        fora_de_linha[:, None], [0.68, 0.24, 0.08],
        np.where(pesados[:, None], [0.62, 0.03, 0.35], [0.85, 0.05, 0.10]),
    )
    sorteio_uso = rng.random(n)
    tipo_uso = np.where(
        sorteio_uso < probs_uso[:, 0], "particular",
        np.where(sorteio_uso < probs_uso[:, 0] + probs_uso[:, 1], "aplicativo", "frota"),
    )
    km_por_ano = np.array([KM_ANUAL_MEDIANO[t] for t in tipo_uso]) * rng.lognormal(0, 0.35, n)
    km_atual = km_por_ano * np.maximum(idade, 0.5) * rng.normal(1, 0.03, n)
    # carro com mais de ~400 mil km quase sempre já saiu de circulação
    km_atual = np.round(np.minimum(km_atual, rng.uniform(380_000, 480_000, n)))

    regiao = rng.choice(REGIOES, size=n, p=PARTICIPACAO_REGIOES)
    distancia = np.array([DISTANCIA_MEDIANA_KM[r] for r in regiao]) * rng.lognormal(0, 0.8, n)
    distancia = np.clip(distancia, 0.5, 400).round(1)

    garantia_estendida = ((idade <= 5) & (rng.random(n) < 0.15)).astype(int)
    garantia_ativa = (idade < 3) | ((garantia_estendida == 1) & (idade < 5))
    prob_plano = np.where(tipo_uso == "frota", 0.30, 0.10) * np.where(idade <= 4, 1.3, 0.6)
    plano_manutencao = (rng.random(n) < prob_plano).astype(int)
    recall_pendente = (rng.random(n) < 0.07).astype(int)
    aceita_whatsapp = (rng.random(n) < 0.65).astype(int)
    idade_cliente = np.where(tipo_uso == "aplicativo", rng.normal(36, 9, n), rng.normal(47, 12, n))
    idade_cliente = np.clip(idade_cliente, 18, 85).round()

    # lealdade é um traço escondido (o CRM não enxerga) que puxa revisões, app, NPS e evasão juntos
    lealdade = rng.normal(0, 1, n)

    media_revisoes = np.exp(
        0.2 + 0.35 * lealdade + 0.35 * garantia_ativa + 0.4 * plano_manutencao
        + 0.3 * (tipo_uso == "frota") - 0.3 * fora_de_linha
    )
    revisoes = np.minimum(rng.poisson(media_revisoes), 8)
    dias = np.where(
        revisoes > 0,
        rng.gamma(2.0, 110 * np.exp(-0.3 * lealdade)) + 15,
        730 + rng.gamma(2.0, 200, n),
    )
    dias = np.clip(dias, 1, 2500).round()

    usa_app = rng.random(n) < _sigmoid(-0.9 + 0.8 * lealdade + 0.5 * (idade < 5) - 0.4 * fora_de_linha)
    sessoes = np.where(usa_app, rng.poisson(np.exp(1.5 + 0.3 * lealdade)) + 1, 0)

    reclamacoes = np.minimum(rng.poisson(0.25 * np.exp(-0.3 * lealdade) + 0.2 * recall_pendente), 5)
    respondeu = rng.random(n) < _sigmoid(0.9 + 0.7 * lealdade)
    nps = np.clip(rng.normal(7.3 + 0.9 * lealdade - 0.8 * reclamacoes, 1.8), 0, 10).round()
    nps = np.where(respondeu, nps, np.nan)

    ticket_mediano = np.array([PERFIL_MODELOS[m][3] for m in modelo]) * (1 + 0.04 * idade)
    ticket_relativo = rng.lognormal(0, 0.3, n)
    ticket = (ticket_mediano * ticket_relativo).round(2)

    sem_garantia = ~garantia_ativa
    logit = (
        1.10 * sem_garantia
        + 0.55 * (np.log1p(distancia) - np.log1p(10))
        + 0.50 * sem_garantia * (distancia > 25)
        + 0.90 * (dias > 365) + 0.35 * (dias / 365)
        - 0.30 * np.minimum(revisoes, 4)
        + 0.85 * (respondeu & (np.nan_to_num(nps, nan=10) <= 6))
        - 0.45 * (respondeu & (np.nan_to_num(nps, nan=0) >= 9))
        + 0.35 * ~respondeu
        + 0.90 * np.log(ticket_relativo)
        + 0.35 * np.minimum(reclamacoes, 3)
        + 0.45 * (tipo_uso == "aplicativo")
        - 0.50 * (tipo_uso == "frota")
        + 0.45 * fora_de_linha
        - 1.00 * plano_manutencao
        - 0.55 * usa_app
        - 0.35 * recall_pendente
        + 0.15 * (regiao == "Norte")
        - 0.60 * lealdade
        + rng.normal(0, 0.6, n)
    )
    intercepto = _calibrar_intercepto(logit, TAXA_EVASAO_ALVO)
    evadiu = (rng.random(n) < _sigmoid(logit + intercepto)).astype(int)

    vins = rng.choice(CARACTERES_VIN, size=(n, 14))
    return pd.DataFrame({
        "id_cliente": [f"CLI-{i:05d}" for i in range(1, n + 1)],
        "vin": ["9BF" + "".join(linha) for linha in vins],
        "data_referencia": DATA_REFERENCIA,
        "modelo": modelo,
        "ano_modelo": ano_modelo,
        "km_atual": km_atual,
        "tipo_uso": tipo_uso,
        "regiao": regiao,
        "distancia_concessionaria_km": distancia,
        "garantia_estendida": garantia_estendida,
        "plano_manutencao": plano_manutencao,
        "revisoes_rede_24m": revisoes,
        "dias_desde_ultima_visita": dias,
        "ticket_medio_rs": ticket,
        "nps_ultima_visita": nps,
        "reclamacoes_12m": reclamacoes,
        "recall_pendente": recall_pendente,
        "sessoes_app_30d": sessoes,
        "aceita_whatsapp": aceita_whatsapp,
        "idade_cliente": idade_cliente,
        "motivo_saida_pesquisa": np.nan,
        TARGET: evadiu,
    })


def _sujar(df, rng):
    """Coloca os erros que um CRM real tem: texto fora do padrão, valores impossíveis, outliers e ausentes."""
    df = df.astype({"modelo": object, "tipo_uso": object, "regiao": object, "motivo_saida_pesquisa": object})
    n = len(df)

    def linhas(fracao, mascara=None):
        sorteio = rng.random(n) < fracao
        if mascara is not None:
            sorteio &= mascara
        return np.flatnonzero(sorteio)

    variantes_modelo = [str.upper, str.lower, lambda s: s + " ", lambda s: " " + s.lower(), lambda s: s.replace(" ", "-")]
    idx = linhas(0.08)
    escolhas = rng.integers(0, len(variantes_modelo), len(idx))
    df.loc[idx, "modelo"] = [variantes_modelo[k](m) for k, m in zip(escolhas, df.loc[idx, "modelo"])]

    variantes_uso = {
        "particular": ["Particular", "PARTICULAR", "PF", "pf"],
        "aplicativo": ["App", "APP", "Aplicativo"],
        "frota": ["Frota", "PJ", "FROTA"],
    }
    idx = linhas(0.10)
    df.loc[idx, "tipo_uso"] = [rng.choice(variantes_uso[t]) for t in df.loc[idx, "tipo_uso"]]

    idx = linhas(0.03)
    df.loc[idx, "regiao"] = [r.upper() if k else r.lower() for k, r in zip(rng.integers(0, 2, len(idx)), df.loc[idx, "regiao"])]

    idx = linhas(0.005)
    df.loc[idx, "km_atual"] = -df.loc[idx, "km_atual"]
    df.loc[linhas(0.004), "km_atual"] = 999_999
    idx = linhas(0.004)
    df.loc[idx, "dias_desde_ultima_visita"] = -rng.integers(1, 90, len(idx))
    idx = linhas(0.006, df["nps_ultima_visita"].notna().to_numpy())
    df.loc[idx, "nps_ultima_visita"] = rng.choice([11, 12, -1, 99], len(idx))
    idx = linhas(0.003)
    df.loc[idx, "revisoes_rede_24m"] = rng.integers(25, 61, len(idx))
    idx = linhas(0.003)
    df.loc[idx, "idade_cliente"] = rng.choice([0, 150, 999], len(idx))
    idx = linhas(0.003)
    df.loc[idx, "distancia_concessionaria_km"] = rng.uniform(1_200, 3_000, len(idx)).round(1)
    idx = linhas(0.008)
    df.loc[idx, "ticket_medio_rs"] = (df.loc[idx, "ticket_medio_rs"] * rng.uniform(8, 25, len(idx))).round(2)

    ausentes = {"idade_cliente": 0.06, "ticket_medio_rs": 0.05, "distancia_concessionaria_km": 0.04, "km_atual": 0.03, "regiao": 0.02}
    for coluna, fracao in ausentes.items():
        df.loc[linhas(fracao), coluna] = np.nan
    return df


def _adicionar_vazamento(df, rng):
    """Só quem já foi embora responde a pesquisa de saída, então essa coluna entrega o alvo."""
    idx = np.flatnonzero((df[TARGET] == 1).to_numpy() & (rng.random(len(df)) < 0.6))
    df.loc[idx, "motivo_saida_pesquisa"] = rng.choice(MOTIVOS_SAIDA, len(idx), p=[0.3, 0.25, 0.15, 0.2, 0.1])
    return df


def _duplicar(df, rng, fracao=0.01):
    """Duplica ~1% das linhas (cliente importado duas vezes) e embaralha a base."""
    copias = df.iloc[rng.choice(len(df), size=round(fracao * len(df)), replace=False)]
    df = pd.concat([df, copias], ignore_index=True)
    return df.iloc[rng.permutation(len(df))].reset_index(drop=True)


def gerar_dados(n=N_CLIENTES, seed=SEED):
    rng = np.random.default_rng(seed)
    df = _gerar_clientes(n, rng)
    df = _sujar(df, rng)
    df = _adicionar_vazamento(df, rng)
    df = _duplicar(df, rng)
    return df.astype({coluna: "Int64" for coluna in COLUNAS_INTEIRAS})[COLUNAS_CSV]


def main():
    df = gerar_dados()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    unicos = df.drop_duplicates(subset="id_cliente")
    print(f"{len(df)} linhas ({len(unicos)} clientes) salvas em {DATA_PATH}")
    print(f"Taxa de evasão: {unicos[TARGET].mean():.1%}")


if __name__ == "__main__":
    main()
