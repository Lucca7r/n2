Resumo do Script
Este script implementa um Algoritmo Genético Híbrido para otimização de problemas mistos (variáveis categóricas e inteiras). Aqui está o que ele faz:

Objetivo Principal
Encontrar a melhor configuração de parâmetros que maximize uma função objetivo, utilizando:

1 variável categórica (nominal): "baixo", "medio" ou "alto"
9 variáveis inteiras: valores entre 1 e 100
Técnicas Utilizadas
Algoritmo Genético (GA):

População de 80 indivíduos
30 gerações
Seleção por torneio (k=3)
Crossover de 1 ponto (taxa 90%)
Mutação com "creep" (pequenos passos) e reset aleatório
Elitismo: mantém os 8 melhores
Busca Local (Pattern Search):

Refinamento intensivo dos 5 melhores indivíduos por geração
Explora vizinhança testando categorias alternativas e pequenas variações nos inteiros
Até 30 avaliações por refinamento
Parada Antecipada:

Para se não houver melhoria por 8 gerações consecutivas
Integração com Simulador
Preparado para chamar um executável modelo10.exe (atualmente desabilitado: USE_SUBPROCESS = False)
Usa função objetivo fictícia para testes quando o executável não está disponível
Saída
Gera arquivo CSV (avaliacoes_log.csv) com histórico completo de todas as avaliações
Retorna a melhor configuração encontrada e seu valor objetivo
Em resumo: É um otimizador robusto que combina exploração global (GA) com refinamento local (PS) para problemas de otimização mista.
