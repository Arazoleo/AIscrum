import pandas as pd
from slack_sdk import WebClient
from jira import JIRA
import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def connectJira(server, username, token):
    try:
        jira = JIRA(server=server, basic_auth=(username, token))
        st.success("Conexão bem sucedida ao Jira")
        return jira
    except Exception as e:
        st.error(f"Erro ao conectar ao Jira: {e}")
        return None


def createTask(jira, project, summary, description, typeT='Task', priority='Medium'):
    try:
        newTask = jira.create_issue(
            project=project,
            summary=summary,
            description=description,
            issuetype={'name': typeT},
        )
        st.success(f"Tarefa: '{summary}' criada com sucesso")
        return newTask
    except Exception as e:
        st.error(f"Erro ao criar tarefa: {e}")


def send_notif(token, channel, message):
    client = WebClient(token=token)
    try:
        client.chat_postMessage(channel=channel, text=message)
        st.success("Notificação enviada para o Slack")
    except Exception as e:
        st.error(f"Erro ao enviar notificação: {e}")


def get_tasks_from_jira(jira, project_key):
    try:
        issues = jira.search_issues(f'project={project_key}')
        tasks = []
        for issue in issues:
            tasks.append({
                'id': issue.id,
                'titulo': issue.fields.summary,
                'data_criacao': issue.fields.created,
                'data_conclusao': issue.fields.resolutiondate,
                'status': issue.fields.status.name,
                'progresso': 100 if issue.fields.status.name == 'Concluída' else 0  
            })
        return tasks
    except Exception as e:
        st.error(f"Erro ao buscar tarefas do Jira: {e}")
        return []


def train_model(data):
    l = LabelEncoder()
    data['prioridade'] = l.fit_transform(data['prioridade'])
    data['tipo'] = l.fit_transform(data['tipo'])
    
    x = data[['prioridade', 'tipo']]
    y = data['tempo_estimado']
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
   
    y_pred = model.predict(X_test)
    
   
    mae = mean_absolute_error(y_test, y_pred)  
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  
    r2 = r2_score(y_test, y_pred)  
    
    
    st.subheader("Resultados do Modelo de Machine Learning")
    st.write(f"Erro Médio Absoluto (MAE): {mae:.2f}")
    st.write(f"Erro Quadrático Médio (RMSE): {rmse:.2f}")
    #st.write(f"Coeficiente de Determinação (R²): {r2:.2f}")
    
    return mae, rmse, r2


def predict_duration(model, prioridade, tipo):
    prio_enco = [prioridade]
    tip_enco = [tipo]
    predict = model.predict(np.array([[prio_enco[0], tip_enco[0]]]))
    return predict[0]


server = 'https://arazoleonardo.atlassian.net'
username = 'arazoleonardo@gmail.com'
token = 'ATATT3xFfGF0qmSprJzUKC8gPpd4FPY1IwXxMAwciVitWXmrFKSgLdQ7-q_te1jruDIMU3ql6ly2cE_0lpvVChSCZCLz5IhzmxDuljWFN5sblBnkqFLJ-0vMDKq7WrmsUmwrCNdzIqmmooRCTfPwS_OYz7IjFQOCzbZTV4S8S6kSS7jcOt4Kzfs=6961FD2D'

jira = connectJira(server, username, token)

historical_data = pd.read_csv('historical_tasks.csv')
model, X_test, y_test = train_model(historical_data)

if jira:
    summ = st.text_input("Título da Tarefa:", "Novo projeto de IA em Python")
    desc = st.text_area("Descrição da Tarefa:", "Criar uma IA usando Pytorch para realizar operações matemáticas complexas")
    pro = 'PROJ'
    
    if st.button("Criar Tarefa"):
        task = createTask(jira, pro, summ, desc)
        
        slack_tok = 'xoxb-8008254179440-7985634820226-hkYR0hvEcefom8fCbrOKzGQj'
        chann = 'projeto-python-scrum'
        messa = f'A nova tarefa foi criada no JIRA: *{summ}* - {desc}'
        send_notif(slack_tok, chann, messa)


if jira:
    project_key = st.text_input("Insira a chave do projeto Jira:", "PROJ")  
    if st.button("Buscar Tarefas"):
        tarefas = get_tasks_from_jira(jira, project_key)
        if tarefas:
            df = pd.DataFrame(tarefas)
            
            st.subheader("Dashboard de Progresso das Tarefas")
            st.line_chart(df.set_index('data_criacao')['progresso'])
    
            st.subheader("Tabela de Tarefas")
            st.table(df)

   
    evaluate_model(model, X_test, y_test)
