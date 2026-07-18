"""Generate original, reviewable packs for the remaining Class 8 Science chapters.

This is deliberately a deterministic content pipeline rather than a scraper. The
official NCERT chapter URL is retained as provenance, while the question text is
generated from a small, inspectable concept manifest. That makes a stage reset
repeatable and keeps the content easy to inspect or replace later.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from uuid import uuid5

from .class8_science_catalog import CHAPTERS, CURRICULUM_VERSION_ID, ChapterSpec, chapters_after
from .models import CandidateOption, QuestionCandidate


@dataclass(frozen=True)
class TopicSeed:
    slug: str
    title: str
    definition: str
    example: str
    distinction: str
    application: str
    correction: str


def topic(
    slug: str,
    title: str,
    definition: str,
    example: str,
    distinction: str,
    application: str,
    correction: str,
) -> TopicSeed:
    return TopicSeed(slug, title, definition, example, distinction, application, correction)


# Ten small concept seeds per chapter keep the pack inspectable and make the
# generated questions easy to replace with reviewed editorial content later.
TOPICS: dict[int, tuple[TopicSeed, ...]] = {
    2: (
        topic("microorganisms", "Microorganisms", "Microorganisms are living things too small to see clearly without magnification.", "Yeast cells can make dough rise.", "A microorganism is defined by its tiny size, not by whether it is useful or harmful.", "Use a microscope to observe a prepared sample rather than guessing from an unaided view.", "Not every microorganism causes disease; many are useful or harmless."),
        topic("microscope", "Microscopes", "A microscope uses lenses to produce a magnified view of a tiny specimen.", "A prepared drop of pond water can reveal tiny moving organisms.", "Magnification makes an image appear larger, while the specimen itself does not become larger.", "Focus the instrument carefully and use a prepared slide to observe a small sample.", "A larger-looking image does not by itself prove that every detail is accurate."),
        topic("cells", "Cells", "A cell is the basic structural and functional unit of living organisms.", "A leaf peel can show many small cells under a microscope.", "Cells are parts of living organisms; a complete organism may contain one cell or many cells.", "Compare cell shapes and structures in prepared plant and animal samples.", "A cell is not the same thing as a whole multicellular organism."),
        topic("unicellular", "Unicellular organisms", "A unicellular organism carries out life processes within a single cell.", "Amoeba performs feeding and movement using one cell.", "Unicellular describes the number of cells, not the organism's size or usefulness.", "Observe a prepared microorganism and identify how one cell performs several functions.", "One cell can be a complete organism; it is not automatically a non-living particle."),
        topic("multicellular", "Multicellular organisms", "A multicellular organism is made of many cells that may have specialised roles.", "Human muscle and nerve cells perform different functions.", "Specialisation allows groups of cells to work together rather than every cell doing exactly the same job.", "Relate a visible body part to the different kinds of cells that support its function.", "Many cells do not mean that each cell can live independently as a complete organism."),
        topic("beneficial-microbes", "Beneficial microorganisms", "Some microorganisms help make food, medicines, or useful products.", "Lactobacillus helps turn milk into curd.", "Beneficial means useful in a particular context; it does not mean the organism is always harmless.", "Use a controlled fermentation example to explain how microbes can change food.", "Calling a microbe beneficial in one process does not mean every microbe is beneficial."),
        topic("harmful-microbes", "Harmful microorganisms", "Some microorganisms cause disease or spoil food by their activities.", "Mould growing on damp bread can spoil it.", "Harm depends on the interaction and context, not simply on the organism being microscopic.", "Keep food dry, covered, and stored correctly to reduce unwanted microbial growth.", "Spoilage and disease are different effects, even though both may involve microorganisms."),
        topic("disease-transmission", "Disease transmission", "Communicable diseases can spread from an infected source to another person through a route of transmission.", "Contaminated water can carry disease-causing organisms.", "The pathogen, source, route, and susceptible host are different parts of transmission.", "Break a likely route by using safe water, hand hygiene, or appropriate isolation.", "Being near an ill person does not prove that transmission occurred by one particular route."),
        topic("decomposition", "Decomposition", "Decomposers break down dead material and return substances to the environment.", "Fungi and bacteria help turn fallen leaves into simpler materials.", "Decomposition recycles matter; it is not the same as a living organism eating a complete meal.", "Maintain a compost system with suitable moisture and air to support decomposition.", "Decomposition does not make matter disappear; it changes and recycles it."),
        topic("safe-observation", "Safe observation of microbes", "Microorganisms should be studied with safe samples, clean equipment, and appropriate supervision.", "A sealed prepared slide is safer than culturing unknown microbes in class.", "Seeing a sample is not a reason to handle or grow it without controls.", "Use prepared slides, wash hands, and follow disposal instructions after an observation.", "Unknown microbial growth should never be opened, tasted, or handled casually."),
    ),
    3: (
        topic("health", "Health", "Health is a state of physical, mental, and social well-being, not merely the absence of disease.", "Rest, supportive relationships, and nutritious food can all affect health.", "Health includes well-being and functioning, while disease is a condition that may disturb them.", "Plan a routine that balances food, sleep, physical activity, and emotional support.", "A person who has no diagnosed disease may still need support for overall well-being."),
        topic("balanced-diet", "Balanced diet", "A balanced diet supplies needed nutrients and energy in suitable proportions.", "A meal can combine grains, pulses, vegetables, fruit, and other locally suitable foods.", "Balanced means varied and adequate, not simply eating a large amount of one food.", "Evaluate a meal by looking for different food groups and reasonable portions.", "A costly or fashionable food is not automatically part of a balanced diet."),
        topic("nutrients", "Nutrients", "Nutrients are substances in food that the body needs for energy, growth, repair, and regulation.", "Carbohydrates can provide energy while proteins support growth and repair.", "Nutrients are components of food; foods can contain more than one nutrient.", "Match a food choice to the nutrient need it can help meet.", "No single nutrient performs every body function."),
        topic("deficiency", "Nutrient deficiency", "A nutrient deficiency occurs when the body does not receive enough of a needed nutrient over time.", "Insufficient iron intake can contribute to anaemia.", "A deficiency is linked to inadequate nutrient supply, not to a food being unfamiliar.", "Use a varied diet and qualified health advice to reduce deficiency risk.", "One symptom alone cannot diagnose a deficiency."),
        topic("communicable-disease", "Communicable diseases", "A communicable disease can spread between people or through contaminated sources.", "A respiratory infection may spread through droplets or close contact.", "Communicable refers to spread; it does not describe how severe every illness will be.", "Choose prevention based on the route, such as ventilation, hygiene, or safe food handling.", "Not every disease is communicable."),
        topic("pathogens", "Pathogens", "Pathogens are disease-causing organisms or infectious agents.", "Some bacteria, viruses, fungi, and parasites can act as pathogens.", "A pathogen is identified by its ability to cause disease, not merely by being microscopic.", "Use reliable public-health guidance to reduce exposure to a known pathogen.", "A microorganism is not automatically a pathogen."),
        topic("immunity", "Immunity and vaccines", "The immune system helps the body recognise and respond to disease-causing agents.", "Vaccination prepares immune memory against a specific disease threat.", "Vaccines support prevention; they are not a treatment for every illness after it begins.", "Follow the recommended vaccination schedule and seek medical guidance when needed.", "Vaccination does not mean a person can ignore every other preventive measure."),
        topic("hygiene", "Hygiene", "Hygiene practices reduce exposure to harmful agents and interrupt routes of infection.", "Washing hands before eating can reduce transfer from hands to food.", "Hygiene reduces risk but cannot guarantee that no one will ever become ill.", "Identify the route of spread and select a hygiene practice that interrupts it.", "Using more soap than necessary does not replace thorough washing."),
        topic("non-communicable-disease", "Non-communicable diseases", "A non-communicable disease does not spread from one person to another as an infection.", "Some long-term heart conditions are influenced by biology and lifestyle rather than person-to-person spread.", "Non-communicable describes transmission, not whether prevention or treatment is possible.", "Use regular activity, nutritious food, and professional advice to manage risk.", "A disease being non-communicable does not make it unimportant."),
        topic("public-health", "Public health", "Public health protects communities through prevention, safe environments, and access to care.", "Clean water systems and vaccination programmes protect many people at once.", "Public health acts at community scale, while personal habits act at individual scale.", "Combine personal hygiene with community measures such as safe water and waste management.", "Individual action alone cannot solve every public-health problem."),
    ),
    4: (
        topic("electric-circuit", "Electric circuits", "An electric circuit is a complete conducting path through which electric current can flow.", "A cell, wires, and a closed switch can light a lamp.", "A closed conducting path is necessary; merely placing components nearby is not enough.", "Trace the complete path from one terminal of the cell and back to the other.", "A circuit with a gap cannot keep the lamp glowing."),
        topic("circuit-components", "Circuit components", "Cells, switches, wires, and loads perform different roles in an electric circuit.", "A switch controls the path while a lamp changes electrical energy into light and heat.", "A source supplies energy, a conductor provides a path, and a load uses energy.", "Represent a circuit using standard symbols before assembling it.", "A switch is not an energy source."),
        topic("conductors-insulators", "Conductors and insulators", "Conductors allow electric current to pass relatively easily, while insulators resist its flow.", "Copper wire conducts, while plastic covering helps insulate it.", "The same object can have conducting and insulating parts for different safety roles.", "Test an unknown material only with a low-voltage classroom circuit.", "A shiny appearance does not by itself prove that a material conducts."),
        topic("heating-effect", "Heating effect of current", "Electric current can produce heat when it passes through resistance in a material.", "An electric heater uses a resistant element to become hot.", "Heating is an energy conversion effect, not evidence that current is being created.", "Select a properly rated appliance and keep flammable materials away from its heating element.", "A hot wire is not automatically safe to touch."),
        topic("magnetic-effect", "Magnetic effect of current", "An electric current can produce a magnetic field around a conductor.", "A current-carrying coil can attract an iron object.", "The magnetic effect comes from current in the circuit, not from the wire simply being made of metal.", "Increase the useful magnetic effect with a coil and a suitable iron core.", "A disconnected wire does not produce the same magnetic effect as a current-carrying one."),
        topic("electromagnet", "Electromagnets", "An electromagnet is a temporary magnet produced by current through a coil, often around an iron core.", "A scrap-yard lifting device can use an electromagnet to pick up iron.", "An electromagnet can be switched and its strength can be changed, unlike a simple permanent magnet.", "Use an insulated coil, a soft iron core, and a low-voltage source for a model.", "An electromagnet should not be connected to a mains supply in a classroom model."),
        topic("electric-motor", "Electric motors", "An electric motor converts electrical energy into mechanical motion using magnetic forces.", "The motor in a toy fan turns its blades.", "A motor produces motion, while a generator uses motion to produce electrical energy.", "Identify the motor as the component that drives a rotating part in an appliance.", "A motor does not run without a suitable circuit and energy source."),
        topic("fuse-safety", "Fuses and electrical safety", "A fuse protects a circuit by melting and opening the path when excessive current flows.", "A fuse can disconnect a circuit during a short circuit or overload.", "A fuse is a safety device, not a switch for routine operation.", "Use the correct fuse rating and never replace a fuse with wire.", "A thicker wire in place of a fuse removes protection rather than improving it."),
        topic("series-parallel", "Series and parallel circuits", "In a series circuit components share one path, while a parallel circuit provides separate branches.", "Two lamps in parallel can continue to receive a path if one branch opens.", "Series and parallel arrangements change how current paths and component behaviour are distributed.", "Choose parallel branches when independent operation of lamps is needed.", "Adding more lamps in series does not usually make every lamp brighter."),
        topic("electrical-energy", "Electrical energy use", "Electrical appliances transfer electrical energy into useful forms such as light, heat, sound, or motion.", "A mixer converts electrical energy mainly into mechanical motion and sound.", "Energy use depends on both appliance power and the time it operates.", "Switch off an appliance when it is not needed and use an efficient option.", "An appliance that is switched on is using energy even if the useful output is small."),
    ),
    5: (
        topic("force", "Force", "A force is a push or pull that can change an object's motion or shape.", "Pushing a box can make it start moving.", "A force is an interaction, so it involves more than simply naming an object.", "Draw the direction of the push or pull when explaining a change in motion.", "A force does not always make an object move if other forces balance it."),
        topic("contact-force", "Contact forces", "A contact force acts when two objects are touching.", "Friction acts between a shoe and the ground during walking.", "Contact is required for forces such as friction and a push by hand.", "Increase or reduce contact friction depending on the task and safety need.", "A contact force cannot act across an empty gap."),
        topic("non-contact-force", "Non-contact forces", "A non-contact force acts without the objects touching.", "Earth's gravity pulls a falling stone downward.", "Gravity, magnetic force, and electrostatic force can act at a distance.", "Predict an interaction even when the objects are separated.", "Lack of touching does not mean that no force is acting."),
        topic("balanced-forces", "Balanced forces", "Balanced forces cancel in their combined effect and do not change an object's motion.", "A book resting on a table has upward support balancing its weight.", "Balanced forces can act on an object that is already moving at constant speed.", "Compare the directions and sizes of forces before deciding whether they are balanced.", "Balanced does not always mean that no forces are present."),
        topic("unbalanced-forces", "Unbalanced forces", "Unbalanced forces have a non-zero combined effect and can change motion.", "A stronger push can make a stationary trolley accelerate.", "An unbalanced force may change speed, direction, or both.", "Look for acceleration or a change in direction as evidence of an unbalanced force.", "A large force is not unbalanced if an equal opposing force cancels it."),
        topic("gravity", "Gravity", "Gravity is an attractive force between masses and gives objects weight near Earth.", "A dropped ball accelerates toward the ground.", "Mass describes the amount of matter, while weight is the gravitational force on that mass.", "Use gravity to explain falling and the support force to explain an object at rest.", "Objects do not fall because the ground pulls them upward."),
        topic("friction", "Friction", "Friction opposes relative motion between surfaces that are in contact.", "Shoe soles need friction with the ground to prevent slipping.", "Friction can be useful for control and harmful when it causes unwanted wear or heating.", "Add tread or roughness when grip is needed and lubrication when rubbing should be reduced.", "Removing all friction would make walking and braking impossible."),
        topic("magnetic-force", "Magnetic force", "Magnetic force can attract or repel certain magnetic materials and magnets without contact.", "A magnet can attract an iron pin through a sheet of paper.", "Magnetic force depends on the materials and poles involved, not on every metal being magnetic.", "Use attraction and repulsion between poles to test a magnet safely.", "All metals are not attracted equally to a magnet."),
        topic("electrostatic-force", "Electrostatic force", "Electrostatic force is an attraction or repulsion between electrically charged objects.", "A rubbed balloon can attract small paper pieces.", "Electrostatic effects involve charge imbalance and can occur without continuous current in a circuit.", "Use a dry, safe demonstration to compare attraction after rubbing different materials.", "An attracted object is not necessarily charged with the opposite sign; polarisation can also matter."),
        topic("force-diagrams", "Force diagrams", "A force diagram represents the important forces on an object with labelled arrows.", "A free-body diagram can show weight downward and support upward on a book.", "Arrow direction and relative size communicate the combined effect of forces.", "Start a force explanation by drawing only the forces relevant to the chosen object.", "A force diagram should not include every object in the room."),
    ),
    6: (
        topic("pressure", "Pressure", "Pressure describes how force is distributed over an area.", "The same force produces more pressure on a sharp tip than on a broad end.", "Pressure depends on force and area, not on force alone.", "Increase area when a design should reduce pressure on a surface.", "A larger contact area does not increase pressure when the force stays the same."),
        topic("area-pressure", "Area and pressure", "For a given force, pressure increases when the contact area decreases.", "Snow shoes spread a person's weight over a larger area.", "Changing area changes pressure even if the total force remains unchanged.", "Use broad foundations or straps to reduce pressure on the supporting surface.", "A broad base does not make the force disappear; it spreads it."),
        topic("liquid-pressure", "Pressure in liquids", "Liquid pressure acts in all directions and generally increases with depth.", "Water escapes farther from a lower hole in a tank than from a higher hole.", "Liquid pressure at a depth is not explained only by the amount of water visible above the container.", "Compare holes at different depths while keeping the liquid and container arrangement controlled.", "Liquid pressure does not act only downward."),
        topic("atmospheric-pressure", "Atmospheric pressure", "Atmospheric pressure is the pressure exerted by the weight of the air around Earth.", "A suction cup holds when outside air pressure and the reduced pressure inside are different.", "Air can exert pressure even though it is not visible.", "Use a carefully controlled demonstration to show that air pressure can support or move objects.", "Air pressure is not absent simply because air cannot be seen."),
        topic("wind", "Wind", "Wind is moving air caused by differences in air pressure.", "Air moves from a region of relatively higher pressure toward lower pressure.", "Wind is air in motion, while pressure difference is one cause that drives it.", "Use a pressure map to predict the broad direction of wind movement.", "Wind does not move randomly with no relationship to pressure differences."),
        topic("convection", "Convection and uneven heating", "Uneven heating can make air move as warmer, less dense air rises and cooler air takes its place.", "Land and sea can heat and cool at different rates near a coast.", "Convection transfers heat through the movement of a fluid such as air or water.", "Explain a local breeze by comparing heating of land, water, and the air above them.", "Warm air rising alone is not the complete explanation; surrounding air must circulate."),
        topic("storms", "Storms", "A storm is a disturbed weather event that may include strong winds, heavy rain, lightning, or thunder.", "Dark clouds, gusty winds, and lightning can signal a developing storm.", "Storm warning signs are observations; they are not a guarantee of the exact path or intensity.", "Seek shelter indoors and avoid exposed locations during a thunderstorm.", "Standing under an isolated tree is not a safe response to lightning."),
        topic("cyclones", "Cyclones", "A cyclone is a large rotating weather system with strong winds around a low-pressure region.", "Warm ocean water can provide energy to a tropical cyclone.", "A cyclone is more than a single gust; it is a large organised weather system.", "Follow official evacuation and shelter instructions when a cyclone warning is issued.", "A calm centre does not mean the whole cyclone has ended."),
        topic("weather-forecasting", "Weather forecasting", "Weather forecasts use observations, instruments, models, and communication to estimate future conditions.", "Pressure, wind, cloud, and satellite observations can support a forecast.", "A forecast is a reasoned prediction with uncertainty, not a perfect guarantee.", "Use the latest official forecast rather than relying on an old message.", "A forecast that changes is not automatically useless; new observations can improve it."),
        topic("cyclone-preparedness", "Cyclone preparedness", "Preparedness reduces harm by planning communication, supplies, shelter, and evacuation before a cyclone.", "A family emergency kit can include water, medicines, a torch, and important contacts.", "Preparedness happens before danger, while response happens during or immediately after it.", "Make a household plan and follow local authorities' instructions.", "Waiting until strong winds arrive is too late to complete every preparation safely."),
    ),
    7: (
        topic("particles", "Particles of matter", "Matter is made of extremely small particles that have spaces and are in continuous motion.", "The smell of perfume can spread through a room.", "The particle model explains observations without requiring particles to be visible directly.", "Use the model to explain diffusion, compression, and changes of state.", "Particles are not necessarily motionless just because a substance appears still."),
        topic("particle-spaces", "Spaces between particles", "Particles of matter have spaces between them, and the amount of space varies between states.", "Air can be compressed more easily than a solid block.", "Compressibility provides evidence of particle spacing rather than proving that matter is empty.", "Compare how air and water respond when gently pressed in suitable syringes.", "Spaces between particles do not mean that a material has no mass."),
        topic("particle-motion", "Motion of particles", "Particles move continuously, and their average motion changes with temperature.", "A drop of ink spreads faster in warm water than in cold water.", "Temperature affects average particle motion; it does not create the particles.", "Compare diffusion under controlled temperature conditions.", "A solid can have particle motion even when its shape remains fixed."),
        topic("solid-state", "Solid state", "In a solid, particles are closely packed and mainly vibrate around fixed positions.", "A metal spoon keeps its shape when placed on a table.", "A solid has a definite shape and volume because its particles are held in an organised arrangement.", "Use particle spacing and motion to explain why a solid resists compression.", "Solid particles are not completely without motion."),
        topic("liquid-state", "Liquid state", "In a liquid, particles remain close but can move past one another, so the liquid flows.", "Water takes the shape of its container while keeping nearly the same volume.", "A liquid has a definite volume but no fixed shape of its own.", "Explain pouring using the ability of liquid particles to move past one another.", "A liquid does not take up no space just because it takes the container's shape."),
        topic("gas-state", "Gaseous state", "In a gas, particles are far apart and move freely, so the gas fills its container.", "Air spreads through an entire room.", "A gas has neither a fixed shape nor a fixed volume under ordinary conditions.", "Use a sealed syringe or balloon to observe that a gas occupies space and can be compressed.", "A gas is not weightless merely because it is hard to see."),
        topic("diffusion", "Diffusion", "Diffusion is the spreading of particles from a region of higher concentration to lower concentration due to random motion.", "A scent spreads from a perfumed area into nearby air.", "Diffusion is a particle process and does not require stirring in every situation.", "Compare diffusion rates while keeping the amount of substance and container size controlled.", "Diffusion does not mean that particles all travel in one straight direction."),
        topic("temperature-particles", "Temperature and particles", "Heating usually increases average particle motion, while cooling usually decreases it.", "Heating a gas can make it expand if pressure is allowed to remain suitable.", "Temperature is related to average particle motion, not simply to how much matter is present.", "Predict how a substance may expand or change state when its temperature changes.", "A larger sample is not necessarily at a higher temperature."),
        topic("change-state", "Changes of state", "Changes of state occur when particle energy and arrangement change while the substance remains the same material.", "Ice melts to liquid water and water vapour can condense back to liquid water.", "A physical change of state does not create a new substance.", "Track heating and cooling to identify melting, freezing, evaporation, and condensation.", "Evaporation can occur at the surface below the boiling point."),
        topic("particle-model-evidence", "Evidence for the particle model", "Observable changes such as diffusion, compression, and expansion support the particle model of matter.", "A sealed balloon changes size when warmed, showing a change in gas behaviour.", "The model is judged by how well it explains several observations together.", "Use more than one observation when evaluating a particle-based explanation.", "One observation alone does not show that every detail of a model is proven."),
    ),
    8: (
        topic("element", "Elements", "An element is a pure substance made of only one kind of atom.", "Copper is an element used in electrical wiring.", "An element cannot be broken into simpler substances by ordinary chemical methods.", "Use a symbol and property list to identify an element in a material sample.", "An element is not defined by whether it is solid, liquid, or gas at room temperature."),
        topic("atom", "Atoms", "An atom is the smallest unit of an element that retains the element's chemical identity.", "A sample of iron contains iron atoms.", "Atoms are units of elements; they are not the same as a visible grain or piece of a substance.", "Use particle diagrams to represent an element as one type of atom.", "An atom is not necessarily the same as a molecule."),
        topic("molecule", "Molecules", "A molecule is a group of atoms held together that behaves as a unit.", "An oxygen molecule contains two oxygen atoms joined together.", "Molecules may contain atoms of one element or of different elements.", "Read a simple molecular model by counting the kinds and numbers of atoms shown.", "Every molecule is not a compound; some molecules contain one element only."),
        topic("compound", "Compounds", "A compound is a pure substance formed when elements combine chemically in a fixed ratio.", "Water contains hydrogen and oxygen chemically combined.", "A compound has properties different from the elements that formed it.", "Distinguish a compound from a mixture by checking whether the substances are chemically combined.", "A compound cannot be separated into its elements by simple physical methods."),
        topic("mixture", "Mixtures", "A mixture contains two or more substances that are physically combined and can often be separated by physical methods.", "Air is a mixture of several gases.", "The substances in a mixture retain their identities and may occur in variable proportions.", "Choose a physical separation method based on a component property.", "A mixture does not necessarily have a fixed composition."),
        topic("physical-properties", "Physical properties", "Physical properties can be observed or measured without changing a substance into a new substance.", "Colour, state, solubility, and density are physical properties.", "A physical property describes a substance without requiring a chemical reaction.", "Compare samples using an agreed property such as solubility or magnetism.", "A property observed once under unknown conditions may not identify a substance conclusively."),
        topic("separation", "Separation of mixtures", "Mixture components can be separated by using differences in physical properties.", "Filtration can separate an insoluble solid from a liquid.", "The method must match the property difference between the components.", "Select filtration, evaporation, sieving, or magnet separation for a particular mixture.", "No single separation method works for every mixture."),
        topic("homogeneous-mixture", "Homogeneous mixtures", "A homogeneous mixture has a uniform composition throughout the sample at the scale being considered.", "Salt solution appears uniform after the salt dissolves.", "Uniform appearance does not make a mixture a compound.", "Take samples from different parts and compare their composition or properties.", "A homogeneous mixture can still contain more than one substance."),
        topic("heterogeneous-mixture", "Heterogeneous mixtures", "A heterogeneous mixture has a non-uniform composition with distinguishable parts or regions.", "Sand in water forms a heterogeneous mixture.", "Different parts can have visibly or measurably different compositions.", "Allow settling or use filtration when components have different observable phases.", "A mixture need not be heterogeneous just because it contains several substances."),
        topic("conservation-purity", "Purity and conservation", "During physical separation, the components of a mixture are recovered without being changed into new substances.", "Evaporating salt water can recover salt while water leaves as vapour.", "Recovering a component is different from destroying or creating matter.", "Record the starting mixture and recovered fractions when evaluating a separation.", "A component that is no longer visible has not necessarily been destroyed."),
    ),
    9: (
        topic("solution", "Solutions", "A solution is a homogeneous mixture in which one or more substances are uniformly distributed in another.", "Salt water is a solution when the salt has dissolved completely.", "A solution is a mixture even though it looks uniform.", "Identify the components of a familiar solution before discussing its concentration.", "A clear liquid is not automatically a solution."),
        topic("solute", "Solute", "The solute is the substance that dissolves in a solution.", "Salt is the solute when it dissolves in water.", "Solute describes the dissolved component, not necessarily the component present in the smaller amount in every case.", "Label the dissolved substance when drawing or preparing a solution.", "The solute does not vanish; its particles spread through the solvent."),
        topic("solvent", "Solvent", "The solvent is the component that dissolves the solute and usually forms the larger part of a solution.", "Water is the solvent in sugar water.", "Solvent refers to the dissolving medium, not to every liquid in a container.", "Choose a solvent based on whether it can dissolve the substance safely.", "Water is a common solvent but not the solvent for every solution."),
        topic("soluble", "Solubility", "Solubility is the ability of a substance to dissolve in a particular solvent under specified conditions.", "Sugar is soluble in water under ordinary conditions.", "Solubility depends on the pair of substances and conditions, not on the solute alone.", "Test equal amounts with a fixed solvent volume and record whether they dissolve.", "A substance being soluble in water does not mean it dissolves in every solvent."),
        topic("insoluble", "Insoluble substances", "An insoluble substance does not dissolve appreciably in a particular solvent under the stated conditions.", "Sand is insoluble in water.", "Insoluble means very little dissolves in that solvent and condition, not that the substance can never dissolve anywhere.", "Use filtration to separate an insoluble solid from water.", "Cloudiness or suspension is not evidence that the solid has dissolved."),
        topic("concentration", "Concentration", "Concentration compares the amount of solute with the amount of solution or solvent used.", "A spoonful of sugar in a small cup is more concentrated than the same spoonful in a large jug.", "Concentration depends on both solute and the chosen volume or mass basis.", "Keep the solvent volume fixed when comparing how much solute is added.", "A larger total volume does not automatically mean a more concentrated solution."),
        topic("saturation", "Saturated solutions", "A saturated solution contains as much dissolved solute as it can under specified conditions.", "Extra salt remains at the bottom after a saturated salt solution is stirred.", "Saturation is a condition at a particular temperature and amount of solvent.", "Add solute gradually and record the point at which undissolved material remains.", "A saturated solution is not necessarily the most concentrated solution possible under all conditions."),
        topic("temperature-solubility", "Temperature and solubility", "Changing temperature can change how much of a substance dissolves, depending on the substance and solvent.", "More sugar generally dissolves in warm water than in the same amount of cold water.", "The effect of temperature is not identical for every solute.", "Compare equal solvent volumes at controlled temperatures and measure the dissolved amount.", "Stirring and temperature are different variables and should not be confused."),
        topic("suspension", "Suspensions", "A suspension contains particles dispersed in a fluid that may settle and can often be filtered.", "Mud in water forms a suspension.", "Suspension particles are not uniformly dissolved like solute particles in a solution.", "Allow settling or filter a suspension to separate its dispersed solid.", "A liquid that looks cloudy is not necessarily a solution."),
        topic("crystallisation", "Crystallisation", "Crystallisation forms solid crystals from a solution by removing some solvent or changing conditions.", "Salt crystals can form when a salt solution loses water by evaporation.", "Crystallisation can recover a dissolved solid that filtration alone cannot catch.", "Evaporate carefully and stop before the recovered crystals are contaminated or overheated.", "A dissolved solute cannot be collected by ordinary filtration before it crystallises."),
    ),
    10: (
        topic("light-travel", "Light and straight-line travel", "Light travels in straight lines through a uniform medium until its path is changed.", "A pinhole can form an image because light rays travel through the small opening.", "A light ray is a model of direction, not a visible thread connecting source and object.", "Use a ray diagram to predict whether an object can be seen from a position.", "Light does not bend simply because a diagram line is drawn at an angle."),
        topic("reflection", "Reflection", "Reflection is the return of light into the original medium after it strikes a surface.", "A mirror reflects light toward an observer.", "Reflection changes direction without requiring the light to pass through the reflecting surface.", "Draw the incident and reflected rays to explain how an object is seen in a mirror.", "A rough surface can reflect light even when it does not form a clear image."),
        topic("angles", "Angles of reflection", "The angle of reflection equals the angle of incidence when both are measured from the normal.", "A ray striking at 30 degrees to the normal reflects at 30 degrees to the normal.", "Angles are measured from the normal line, not from the mirror surface.", "Add a normal at the point of incidence before measuring either angle.", "Using the mirror surface as the reference gives the wrong angle relationship."),
        topic("plane-mirror", "Plane mirrors", "A plane mirror forms a virtual, upright image of the same size as the object at an equal distance behind the mirror.", "A person sees an upright image in a flat dressing mirror.", "The image is virtual because reflected rays only appear to come from behind the mirror.", "Use the equal-distance rule to locate a plane-mirror image.", "A plane-mirror image is not formed on a screen placed behind the mirror."),
        topic("lateral-inversion", "Lateral inversion", "Lateral inversion is the apparent left-right reversal seen in a plane mirror.", "The word ambulance is printed reversed on the front of a vehicle so it reads correctly in mirrors.", "The mirror does not swap up and down; it reverses the apparent side-to-side orientation.", "Compare a labelled object and its mirror image to identify the reversed side.", "A mirror image is not a simple rotation of the object."),
        topic("spherical-mirrors", "Spherical mirrors", "Spherical mirrors are curved mirror surfaces that can be concave or convex.", "A concave mirror can converge parallel rays near its principal focus.", "Concave and convex shapes produce different ray behaviour and image possibilities.", "Use a principal axis and representative rays when drawing a spherical-mirror diagram.", "A curved mirror cannot always be treated like a plane mirror."),
        topic("refraction", "Refraction", "Refraction is the change in direction of light when it passes between transparent media because its speed changes.", "A pencil appears bent where it enters water.", "The apparent bend occurs at a boundary between media, not because the pencil itself changes shape.", "Draw the ray before and after the boundary when explaining an apparent shift.", "Refraction does not mean that light is reflected back from every boundary."),
        topic("lenses", "Lenses", "A lens is a transparent curved object that refracts light to converge or diverge rays.", "A convex lens can focus sunlight to a bright spot.", "A convex lens generally converges parallel rays, while a concave lens generally diverges them.", "Use a lens to form or enlarge an image only with safe light levels and proper supervision.", "A lens does not magnify every object in every position."),
        topic("image-formation", "Image formation", "An image forms where light rays from an object meet or appear to meet after reflection or refraction.", "A convex lens can form an image on a screen when the object is placed suitably.", "Real images can be caught on a screen; virtual images cannot be caught directly on a screen.", "Move the screen until the rays converge to test for a real image.", "A sharp image seen in a mirror is not automatically a real image."),
        topic("optical-safety", "Optical safety", "Safe light investigations control brightness, distance, and exposure to protect eyes and skin.", "A classroom light box is safer to inspect than looking directly at the Sun.", "Studying light does not require staring at an intense source.", "Use screens, indirect viewing, and teacher-approved equipment during optics experiments.", "Never look directly at the Sun through a lens, mirror, or optical instrument."),
    ),
    11: (
        topic("apparent-motion", "Apparent motion in the sky", "Apparent motion is the change in position seen from an observer and may result from Earth's motion.", "The Sun appears to move across the sky during a day.", "Apparent motion describes what an observer sees, not necessarily the object's actual path around the observer.", "Record sky positions from the same place and time reference before explaining a pattern.", "The Sun's daily apparent motion does not mean it circles Earth each day."),
        topic("earth-rotation", "Earth's rotation", "Earth's rotation about its axis causes the regular cycle of day and night.", "A location faces the Sun during its day and turns away during its night.", "Rotation is spinning about an axis, while revolution is motion around another body.", "Use a globe and lamp model to connect rotation with changing daylight.", "Day and night are not caused by the Sun switching on and off."),
        topic("moon-phases", "Phases of the Moon", "Moon phases are changing appearances of the sunlit part of the Moon seen from Earth.", "The Moon can appear crescent, half, gibbous, or nearly full during a cycle.", "The Moon does not make its own visible light; the visible phase depends on geometry.", "Use a lamp and ball model to explain why the visible bright portion changes.", "Moon phases are not caused simply by Earth's shadow each night."),
        topic("lunar-month", "Lunar cycle", "The repeating sequence of Moon phases follows the Moon's changing position relative to Earth and Sun.", "A new-moon to new-moon cycle is about one month.", "A phase cycle is a pattern of illumination, not a measure of the Moon producing light.", "Observe the Moon on several dates and record phase and direction.", "The Moon does not show the same phase every night."),
        topic("sun-shadows", "Shadows and the Sun", "The length and direction of a shadow change as the Sun's apparent position changes.", "A vertical stick has a shorter shadow near midday than in the morning.", "The shadow is evidence about light direction and angle, not a clock by itself.", "Mark a shadow tip at fixed intervals using the same stick and ground point.", "A shadow length reading is not useful without recording the time and conditions."),
        topic("sundials", "Sundials and time", "A sundial estimates time from the changing direction or length of a shadow cast by the Sun.", "The shadow of a fixed pointer moves across marked positions during the day.", "A sundial needs a fixed setup and local calibration to be useful.", "Calibrate marked shadow positions against a reliable clock on a clear day.", "A sundial cannot give a reliable reading indoors without sunlight."),
        topic("calendars", "Calendars", "Calendars organise time using repeating cycles such as days, months, and years.", "A year-based calendar helps plan seasons, school terms, and festivals.", "A calendar is a human system for organising cycles; it is not the same as observing one sky event.", "Compare a calendar date with a recorded astronomical observation.", "A month is not exactly the same length in every calendar system."),
        topic("sky-observation", "Systematic sky observation", "Systematic observation records the same features at planned times so patterns can be compared.", "A student records the Moon's phase, direction, and time every evening.", "A repeated observation is stronger evidence of a pattern than one unrecorded sighting.", "Use a table with date, time, position, weather, and appearance.", "Memory alone is not a reliable record of a changing sky pattern."),
        topic("seasons", "Seasonal cycles", "Seasonal patterns are linked to Earth's yearly revolution and the tilt of its axis, which change sunlight received by regions.", "Day length and heating vary through the year in many regions.", "Seasons are not caused mainly by Earth being much closer to the Sun in summer.", "Compare sunlight angle and day length between seasons for a location.", "The same season does not occur at the same time in both hemispheres."),
        topic("evidence-sky", "Evidence from sky cycles", "Repeated sky observations can provide evidence for regular cycles and help improve time-keeping.", "A repeated shadow pattern can identify an approximate time of day.", "Evidence supports a model when observations are repeated and agree with its predictions.", "State the observation, the predicted pattern, and the uncertainty separately.", "A regular pattern does not mean every observation will be identical."),
    ),
    12: (
        topic("ecosystem", "Ecosystems", "An ecosystem includes living organisms and the non-living surroundings with which they interact.", "A pond includes organisms, water, light, soil, and dissolved substances.", "An ecosystem is a system of interactions, not just a list of species.", "Map living and non-living components before explaining a change in a local habitat.", "Removing one visible organism does not remove every part of an ecosystem."),
        topic("habitat", "Habitats", "A habitat is the place and conditions that provide an organism with resources and shelter.", "A mangrove habitat provides shallow water, mud, and shelter for particular organisms.", "A habitat is defined by conditions and resources, while a niche describes a role within them.", "Compare how two organisms use different resources in one habitat.", "The entire Earth is not one identical habitat for every organism."),
        topic("producers", "Producers", "Producers make organic food from simpler substances, usually using light energy.", "Green plants make food through photosynthesis.", "Producers supply stored chemical energy to many food relationships; they do not obtain all energy by eating.", "Identify the producer when drawing a food chain in a grassland.", "A producer can still depend on soil, water, air, and other ecosystem conditions."),
        topic("consumers", "Consumers", "Consumers obtain energy and materials by feeding on other organisms.", "A deer obtains food by eating plants.", "A consumer may be a herbivore, carnivore, omnivore, or another feeding type.", "Classify an organism by what it eats and where it gets energy.", "Consumers are not necessarily harmful to the ecosystem."),
        topic("decomposers", "Decomposers", "Decomposers break down dead organisms and wastes and return materials to the environment.", "Fungi and many bacteria decompose fallen leaves.", "Decomposers support matter recycling even though they do not make food like producers.", "Include decomposers when explaining what happens to dead plant material.", "Removing decomposers would interrupt recycling rather than make the ecosystem cleaner."),
        topic("food-chain", "Food chains", "A food chain is a simple representation of how food and energy pass from one organism to another.", "Grass to grasshopper to frog shows one feeding sequence.", "Arrows in a food chain point in the direction of energy transfer from food to eater.", "Start a chain with a producer and label each organism's feeding relationship.", "The arrow should not point from eater to the food it consumes."),
        topic("food-web", "Food webs", "A food web links several food chains and shows that organisms can have multiple feeding relationships.", "A bird may eat insects and seeds while also being prey for another animal.", "A food web represents interconnected paths rather than one isolated chain.", "Predict how a change in one population may affect connected populations.", "Removing one species need not affect only one other species."),
        topic("interdependence", "Interdependence", "Interdependence means organisms and ecosystem components rely on one another in different ways.", "Plants depend on pollinators while pollinators obtain food from flowers.", "Dependence can involve food, shelter, pollination, decomposition, or physical conditions.", "Trace at least two links before predicting the effect of a habitat change.", "Interdependence does not mean every organism depends on every other organism equally."),
        topic("biodiversity", "Biodiversity", "Biodiversity is the variety of living organisms and the variation among them in a place or on Earth.", "A diverse forest may contain many plant, animal, and microbial species.", "Biodiversity includes variety, not simply the number of individual organisms.", "Record different kinds of organisms rather than counting only one common species.", "A habitat with many individuals of one species is not necessarily highly diverse."),
        topic("conservation", "Balance and conservation", "Conservation protects organisms, habitats, and the processes that keep ecosystems functioning.", "Restoring native vegetation can provide food and shelter for local species.", "Conservation aims for long-term functioning rather than only protecting one visible animal.", "Choose a local action that reduces habitat loss or restores a useful ecosystem process.", "Moving any non-native organism into a habitat is not automatically conservation."),
    ),
    13: (
        topic("habitability", "Habitability", "Habitability is the ability of a place to provide conditions that can support life.", "Earth has liquid water, a suitable atmosphere, and a range of temperatures for life.", "Habitability refers to conditions for life, not to a guarantee that life exists everywhere those conditions occur.", "Compare a planet or moon using water, energy, temperature, and atmosphere evidence.", "A place with one favourable condition is not automatically habitable for all life."),
        topic("earth-layers", "Earth's layers", "Earth has interacting layers such as the atmosphere, hydrosphere, geosphere, and biosphere.", "Rocks, air, water, and living organisms form connected Earth systems.", "The layers are systems that interact, not completely isolated shells.", "Trace how a change in one Earth system can affect another.", "The biosphere is not separate from air, water, and rocks."),
        topic("atmosphere", "Atmosphere", "The atmosphere is the layer of gases surrounding Earth and supports weather and life processes.", "The atmosphere supplies gases and helps regulate temperature.", "The atmosphere is a moving, changing system rather than an empty boundary around Earth.", "Use atmospheric observations to connect weather with a life-supporting environment.", "Air pollution can affect the atmosphere even when the sky looks clear."),
        topic("water-cycle", "Water cycle", "The water cycle moves water among the surface, underground, atmosphere, and living organisms.", "Evaporation, condensation, precipitation, and collection form connected processes.", "Water changes location and state through a cycle; it is not used only once.", "Mark the process that moves water from one reservoir to another in a diagram.", "Rainfall does not mean that the water cycle has stopped at the ground."),
        topic("liquid-water", "Liquid water", "Liquid water is important for life because it provides a medium for chemical processes and transport.", "Cells use water to transport substances and support reactions.", "Water being essential does not mean every water source is safe to drink.", "Protect freshwater from contamination and use suitable treatment before drinking.", "Clear water can still contain harmful dissolved substances or microbes."),
        topic("sun-energy", "Sunlight and energy", "Sunlight is the main energy input driving many Earth processes and most food production.", "Plants capture light energy during photosynthesis.", "Energy flows through ecosystems while many materials cycle through Earth systems.", "Connect sunlight, plants, and food energy in an ecosystem explanation.", "The Sun supplies energy but does not replace the need for nutrients and water."),
        topic("greenhouse", "Greenhouse effect", "The greenhouse effect is warming caused when atmospheric gases absorb and re-emit some outgoing heat.", "Water vapour and carbon dioxide contribute to Earth's natural heat balance.", "The natural greenhouse effect supports life; enhanced warming is a change in its intensity.", "Distinguish natural heat retention from human activities that increase greenhouse gas concentrations.", "The greenhouse effect is not the same as ozone-layer damage."),
        topic("resources", "Earth resources", "Earth resources are materials and energy sources that people and other organisms use.", "Freshwater, soil, minerals, forests, and sunlight are resources with different renewal rates.", "Renewable does not mean unlimited or impossible to damage.", "Plan use around availability, regeneration, and ecological impact.", "A resource can be renewable yet become scarce when used faster than it recovers."),
        topic("human-impact", "Human impact", "Human activities can alter Earth systems through pollution, land-use change, extraction, and emissions.", "Removing vegetation can increase erosion and change local water flow.", "An impact may spread through linked systems rather than staying at the original site.", "Identify the cause, affected Earth systems, and a practical mitigation step.", "An impact is not harmless merely because it is local at first."),
        topic("sustainability", "Sustainability", "Sustainability means meeting present needs while maintaining the ability of future generations and ecosystems to meet theirs.", "Reducing waste, reusing materials, and protecting water can support sustainability.", "Sustainability balances environmental, social, and resource considerations over time.", "Compare choices by their full life-cycle impacts rather than one immediate benefit.", "A convenient choice is not automatically sustainable just because it is recyclable."),
    ),
}


FIELD_FOR_TEMPLATE: tuple[str, ...] = (
    "definition",
    "example",
    "distinction",
    "application",
    "correction",
    "definition",
    "example",
    "application",
    "distinction",
    "application",
)


def _make_options(correct: str, distractors: Iterable[str], rotation: int) -> tuple[list[CandidateOption], str]:
    choices = [correct, *distractors]
    if len(set(choice.casefold() for choice in choices)) != 4:
        raise ValueError(f"Generated duplicate option: {choices}")
    position = rotation % 4
    choices[0], choices[position] = choices[position], choices[0]
    ids = ["A", "B", "C", "D"]
    return [CandidateOption(id=ids[index], text=value) for index, value in enumerate(choices)], ids[position]


def _cognitive_level(topic_index: int, template: int) -> str:
    if topic_index < 5:
        return ("recall", "recall", "recall", "understand", "understand", "understand", "apply", "apply", "apply", "apply")[template]
    return ("recall", "recall", "recall", "understand", "understand", "understand", "understand", "apply", "analyze", "analyze")[template]


def _question_style(template: int) -> str:
    if template == 8:
        return "data"
    if template in {6, 9}:
        return "experiment"
    if template in {3, 4, 5, 7}:
        return "scenario"
    return "direct"


def _stem(spec: ChapterSpec, seed: TopicSeed, template: int) -> str:
    if template == 0:
        return f"Which statement best describes {seed.title.lower()} in {spec.title.lower()}?"
    if template == 1:
        return f"Which example is most directly associated with {seed.title.lower()}?"
    if template == 2:
        return f"Which statement best distinguishes {seed.title.lower()} from a nearby idea?"
    if template == 3:
        return f"Which situation best applies the idea of {seed.title.lower()}?"
    if template == 4:
        return f"Which statement corrects a common misunderstanding about {seed.title.lower()}?"
    if template == 5:
        return f"A learner is explaining {seed.title.lower()}. Which description is most accurate?"
    if template == 6:
        return f"Which observation would provide useful evidence about {seed.title.lower()}?"
    if template == 7:
        return f"Which action is the most appropriate use of the idea of {seed.title.lower()}?"
    if template == 8:
        return f"A class records an observation related to {seed.title.lower()}. Which interpretation is best supported?"
    return f"Which plan would best investigate or demonstrate {seed.title.lower()}?"


def _misconception(seed: TopicSeed) -> list[str]:
    return [
        f"confuses_{seed.slug}",
        "uses_one_observation_as_proof",
    ]


def make_question(spec: ChapterSpec, index: int, topic_index: int, seed: TopicSeed, template: int) -> QuestionCandidate:
    answer_field = FIELD_FOR_TEMPLATE[template]
    correct = getattr(seed, answer_field)
    other_seeds = [TOPICS[spec.sequence_number][(topic_index + offset) % 10] for offset in (1, 2, 3)]
    distractors = [getattr(other, answer_field) for other in other_seeds]
    options, correct_id = _make_options(correct, distractors, index)
    difficulty = "easy" if index <= 40 else "medium" if index <= 80 else "hard"
    question_style = _question_style(template)
    return QuestionCandidate(
        chapter_id=spec.id,
        concept_id=uuid5(spec.id, f"concept:{seed.slug}"),
        question_text=_stem(spec, seed, template),
        options=options,
        correct_option_id=correct_id,
        explanation=correct,
        difficulty=difficulty,
        cognitive_level=_cognitive_level(topic_index, template),
        skill_tags=[seed.slug, question_style],
        misconception_tags=_misconception(seed),
        question_style=question_style,
        estimated_time_seconds=45 if difficulty == "easy" else 60 if difficulty == "medium" else 75,
        marks=1,
        source_url=spec.source_url,
        source_locator=f"Chapter {spec.sequence_number}; original question derived from the chapter concept manifest",
        source_question_id=f"class8-science-{spec.sequence_number:02d}-{seed.slug}-{template + 1:02d}",
        question_family_key=f"class8-science-{spec.sequence_number:02d}:{seed.slug}",
        variant_key=f"template-{template + 1:02d}",
        answer_spec={"type": "mcq_single", "correct_option_id": correct_id},
        generation_spec={
            "generator": "class8-science-topic-template-v1",
            "chapter_sequence": spec.sequence_number,
            "topic_slug": seed.slug,
            "template": template,
        },
        license_status="original_question_source_reference_only",
        review_status="approved",
    )


def build_pack(spec: ChapterSpec) -> list[QuestionCandidate]:
    if spec.sequence_number == 1:
        raise ValueError("Chapter 1 has its own reviewed pilot generator.")
    seeds = TOPICS.get(spec.sequence_number)
    if not seeds or len(seeds) != 10:
        raise ValueError(f"Chapter {spec.sequence_number} must have exactly 10 topic seeds.")
    return [
        make_question(spec, index, topic_index, seed, template)
        for index, (topic_index, seed, template) in enumerate(
            ((topic_index, seed, template) for topic_index, seed in enumerate(seeds) for template in range(10)),
            start=1,
        )
    ]


def write_pack(spec: ChapterSpec, questions: list[QuestionCandidate]) -> Path:
    payload = {
        "curriculum_version_id": str(CURRICULUM_VERSION_ID),
        "chapter_id": str(spec.id),
        "source_url": spec.source_url,
        "chapter_title": spec.title,
        "generator": "class8-science-topic-template-v1",
        "questions": [question.model_dump(mode="json") for question in questions],
    }
    spec.pack_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return spec.pack_path


def generate(chapters: Iterable[ChapterSpec] | None = None) -> list[Path]:
    selected = tuple(chapters or chapters_after())
    return [write_pack(spec, build_pack(spec)) for spec in selected]


def _parse_chapters(value: str) -> tuple[ChapterSpec, ...]:
    selected: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    return tuple(next(chapter for chapter in CHAPTERS if chapter.sequence_number == number) for number in sorted(selected))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Class 8 Science packs for Chapters 2–13.")
    parser.add_argument("--chapters", default="2-13", help="Comma-separated chapter numbers or ranges, for example 2-5,8")
    args = parser.parse_args()
    paths = generate(_parse_chapters(args.chapters))
    print(json.dumps({"generated": [str(path) for path in paths]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
