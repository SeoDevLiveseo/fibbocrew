from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.knowledge.source.csv_knowledge_source import CSVKnowledgeSource

# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


# Create a CSV knowledge source
csv_source = CSVKnowledgeSource(
    file_paths=["cliente-uniasselvi_.csv"]
)


@CrewBase
class Fibboproject():
	"""Fibboproject crew"""

	# Learn more about YAML configuration files here:
	# Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
	# Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	# If you would like to add tools to your agents, you can learn more about it here:
	# https://docs.crewai.com/concepts/agents#agent-tools
	@agent
	def consultor(self) -> Agent:
		return Agent(
			config=self.agents_config['consultor'],
			verbose=True
		)

	@agent
	def mensageiro(self) -> Agent:
		return Agent(
			config=self.agents_config['mensageiro'],
			verbose=True
		)

	# To learn more about structured task outputs, 
	# task dependencies, and task callbacks, check out the documentation:
	# https://docs.crewai.com/concepts/tasks#overview-of-a-task
	@task
	def buscar_e_filtrar_dados(self) -> Task:
		return Task(
			config=self.tasks_config['buscar_e_filtrar_dados'],
		)

	@task
	def interpretar_e_criar_mensagem(self) -> Task:
		return Task(
			config=self.tasks_config['interpretar_e_criar_mensagem'],
			output_file='report.md'
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the Fibboproject crew"""
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			knowledge_sources=[csv_source],
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
			embedder={
				"provider": "google",
				"config": {
					"model": "models/text-embedding-004",
					"api_key": "AIzaSyD1eQjIDg8Y7i7Xr5w89DatopH6o5aP5PI",
				}
			}
		)
