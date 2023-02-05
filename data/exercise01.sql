select id, country, age, workclass_name, education_level, education_num, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_week, over_50k
from (
	select records.id, records.age, workclasses.name as workclass_name, education_levels.name as education_level, education_num, marital_statuses.name as marital_status, occupations.name as occupation, relationships.name as relationship, races.name as race, sexes.name as sex, capital_gain, capital_loss, hours_week, countries.name as country, over_50k
	from records
	left join workclasses
		on records.workclass_id = workclasses.id
	left join education_levels
		on records.education_level_id = education_levels.id
	left join marital_statuses
		on records.marital_status_id = marital_statuses.id
	left join occupations
		on records.occupation_id = occupations.id
	left join relationships
		on records.relationship_id = relationships.id
	left join races
		on records.race_id = races.id
	left join sexes
		on records.sex_id = sexes.id
	left join countries
		on records.country_id = countries.id
);