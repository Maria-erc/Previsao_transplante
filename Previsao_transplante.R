# Análise de Dados da Saúde

# Projeto - É possível prever o tempo de sobrevivência de um paciente com transplante de fígado após um ano do transplante?

# Diretório
setwd('C:/Users/madu_/OneDrive/Documents/DSA_Data_Science/BA/Cap09')
getwd()

# Instala pacotes
install.packages('dplyr')
install.packages('ggcorrplot')
install.packages('forecast')
install.packages('nnet')
install.packages('neuralnet')

# Pacotes
library(dplyr)
library(ggcorrplot)
library(forecast)
library(nnet)
library(neuralnet)

# Importa dados
dados = read.csv('dataset.csv', header = TRUE, na.strings = c(''))
dim(dados) #79100,46
str(dados)

### ANÁLISE DESCRITIVA ###
## Transforma tipos dos dados

# Verifica distribuição das variáveis
hist(dados$AGE) # idade do paciente
hist(dados$AGE_DON) # idade do doador
hist(dados$PTIME) # tempo de sobrevivência após transplante (em dias)
hist(dados$DAYSWAIT_CHRON)# dias de esperado transplante
hist(dados$FINAL_MELD_SCORE) # escala meld

# Variáveis categóricas
dados$DIAB = as.factor(dados$DIAB) # grau de dabetes do paciente
table(dados$DIAB)

dados$PSTATUS = as.factor(dados$PSTATUS)
table(dados$PSTATUS)

dados$GENDER = as.factor(dados$GENDER)
dados$GENDER_DON = as.factor(dados$GENDER_DON)
table(dados$GENDER)
table(dados$GENDER_DON)

dados$REGION = as.factor(dados$REGION)
table(dados$REGION)

dados$TX_Year = as.factor(dados$TX_Year)
table(dados$TX_Year)

dados$MALIG = as.factor(dados$MALIG)
table(dados$MALIG)
barplot(table(dados$MALIG))

barplot(table(dados$ALCOHOL_HEAVY_DON))
sum(is.na(dados$ALCOHOL_HEAVY_DON))
table(dados$ALCOHOL_HEAVY_DON)
View(dados[is.na(dados$ALCOHOL_HEAVY_DON), ])
dados$ALCOHOL_HEAVY_DON <- ifelse(is.na(dados$ALCOHOL_HEAVY_DON), "U", dados$ALCOHOL_HEAVY_DON)

dados$HIST_CANCER_DON = as.factor(dados$HIST_CANCER_DON)
table(dados$HIST_CANCER_DON)

# Identifica todas as colunas do tipo chr
colunas_chr <- sapply(dados, is.character)

# Transformar apenas as colunas do tipo chr em fator
dados[colunas_chr] <- lapply(dados[colunas_chr], as.factor)

## Separa os dataframes
# Pacientes que sobreviveram ao primeiro ano após cirurgia
dados1 = subset(dados, PTIME > 365)
dados1$PTIME = dados1$PTIME - 365
dim(dados1)

# Pacientes que sobreviveram ao primerio ano após cirurgia e que sbreviveram por até 3 anos após a cirurgia
dados2 = subset(dados1, PTIME > 1095)
barplot(dados2$PTIME)

# Separa variáveis numéricas
dados_num = dados2[,!unlist(lapply(dados2, is.factor))]
dim(dados_num) # 25 colunas
str(dados_num)

dados_fator = dados2[,unlist(lapply(dados2, is.factor))]
dim(dados_fator) # 21 colunas

## Calcula correlação
# Correlação das variáveis numéricas (para nominais é associação)
df_corr = round(cor(dados_num, use = 'complete.obs'),2)
ggcorrplot(df_corr)

## Padronização antes da divisão de treino e teste
# Padroniza variáveis numéricas
dados_num_norm = scale(dados_num)

# Combina com variáveis categóricas para um novo df
dados_final = cbind(dados_num_norm, dados_fator)
hist(dados_final$PTIME)
dim(dados_final)
View(dados_final)

# Divisão dos dados em treino e teste
# Definindo diferentes sementes e escolhendo números do vetor (70% treino, 30% teste)
set.seed(1)
index = sample(1:nrow(dados_final), dim(dados_final)[1]*.7)
dados_treino = dados_final[index,]
dados_teste = dados_final[-index,]
dim(dados_treino)
dim(dados_teste)

# Elimina anos 2001 e 2002 por causa dos poucos regristros
dados_treino = subset(dados_treino, !TX_Year %in% c(2001, 2002))
dados_teste = subset(dados_teste, !TX_Year %in% c(2001, 2002))
table(dados_treino$TX_Year)
table(dados_teste$TX_Year)

### MODELO DE REGRESSÃO

?lm

modelo_v1 <- lm(PTIME ~ FINAL_MELD_SCORE + 
                  REGION + 
                  LiverSize + 
                  LiverSizeDon +
                  ALCOHOL_HEAVY_DON +
                  MALIG + 
                  TX_Year,
                data = dados_treino)


summary(modelo_v1)

## Avaliação do modelo
# Com dados de treino
modelo_v1_pred_1 = predict(modelo_v1, new_data = dados_treino)
length(modelo_v1_pred_1)
length(dados_treino$PTIME)
accuracy(modelo_v1_pred_1, dados_treino$PTIME)

# Com dados de teste
modelo_v1_pred_2 = predict(modelo_v1, newdata = dados_teste)
accuracy(modelo_v1_pred_2, dados_teste$PTIME)

# Distribuição do erro de validação
par(mfrow = c(1,1)) # área de plotagem
residuos = dados_teste$PTIME - modelo_v1_pred_2
hist(residuos, xlab = 'Resíduos', main = 'Sobreviventes de 1 a 3 anos')

## Padronização depois da divisão de treino e teste
set.seed(1)
index = sample(1:nrow(dados2), dim(dados2)[1] * .7)
dados_treino = dados2[index,]
dados_teste = dados2[-index,]

# Separa variáveis numéricas e categóricas
# Treino
dados_treino_num = dados_treino[, !unlist(lapply(dados_treino, is.factor))]
dim(dados_treino_num)

dados_treino_fator = dados_treino[,unlist(lapply(dados_treino, is.factor))]
dim(dados_treino_fator)

# Teste
dados_teste_num = dados_teste[, !unlist(lapply(dados_teste, is.factor))]
dim(dados_teste_num)

dados_teste_fator = dados_teste[, unlist(lapply(dados_teste, is.factor))]
dim(dados_teste_fator)

# Padronização treino
dados_treino_num_norm = scale(dados_treino_num)
dados_treino_final = cbind(dados_treino_num_norm, dados_treino_fator)
dim(dados_treino_final)

# Padronização teste
dados_teste_num_norm = scale(dados_teste_num)
dados_teste_final = cbind(dados_teste_num_norm, dados_teste_fator)
dim(dados_teste_final)

# Elimina anos 2001 e 2002 por causa dos poucos regristros
dados_treino_final = subset(dados_treino_final, !TX_Year %in% c(2001, 2002))
dados_teste_final = subset(dados_teste_final, !TX_Year %in% c(2001, 2002))

# Modelo de Regressão Linear
modelo_v1 <- lm(PTIME ~ FINAL_MELD_SCORE + 
                  REGION + 
                  LiverSize + 
                  LiverSizeDon + 
                  ALCOHOL_HEAVY_DON + 
                  MALIG + 
                  TX_Year,
                data = dados_treino_final)

summary(modelo_v1)

# Avaliação do modelo
# Com dados do treino
modelo_v1_pred_1 = predict(modelo_v1, newdata = dados_treino_final)
accuracy(modelo_v1_pred_1, dados_treino_final$PTIME)

# Com dados de teste
modelo_v1_pred_2 = predict(modelo_v1, newdata = dados_teste_final)
accuracy(modelo_v1_pred_2, dados_teste_final$PTIME)

# Distribuição do erro de validação
par(mfrow= c(1,1))
residuos = dados_teste_final$PTIME - modelo_v1_pred_2
hist(residuos, xlab = 'Resíduos', main = 'Sobreviventes de 1 a 3 anos')

# Desfazer a escala dos dados
variaveis_amostra = c("PTIME",
                       "FINAL_MELD_SCORE",
                       "REGION",
                       "LiverSize",
                       "LiverSizeDon",
                       "ALCOHOL_HEAVY_DON",
                       "MALIG",
                       "TX_Year")

# Remove valores NA das variaveis que vai usar no unscale
dados_unscale = na.omit(dados2[,variaveis_amostra]) #quando cria em lm, o moldeo tira os valores NA
dim(dados_unscale)

dados_final_unscale = dados_unscale[-index,]
dim(dados_final_unscale)

dados_final_unscale = subset(dados_final_unscale, !TX_Year %in% c(2001, 2002))

# Histograma de dados sem escala (formato original)
previsoes = predict(modelo_v1, newdata = dados_final_unscale)
hist(previsoes)
accuracy(dados_final_unscale$PTIME, previsoes)

