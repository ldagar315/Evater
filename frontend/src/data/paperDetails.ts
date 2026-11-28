import { chapterPracticeHref } from '../lib/chapterPracticeTarget'
import { samplePapers, type SamplePaper } from './samplePapers'

export interface PaperSection {
  name: string
  marks: number
  questions: number
  detail: string
}

export interface PaperChapterSample {
  chapter: string
  questions: number
}

export interface PaperDetail {
  /** Matches SamplePaper.id exactly, one detail page per paper. */
  paperId: string
  /** URL slug inside /papers/. Lowercase, hyphenated, no trailing slash. */
  slug: string
  /** Canonical path including the trailing slash used by prerendered directory indexes. */
  path: string
  heading: string
  seoTitle: string
  seoDescription: string
  intro: string
  /** What the actual PDF asks, grounded in its questions. Original summaries, not copied stems. */
  inside: string[]
  sections: PaperSection[]
  answersNote: string
  /** Chapter or concept reference exactly as the PDF states it, plus the originality note. */
  sourceNote: string
  /** A live study guide for the same chapter, when one exists. */
  studyGuide?: { label: string; href: string }
  /** Mixed papers only: how the questions sample the chapters. */
  chapterSample?: PaperChapterSample[]
}

const CHAPTER_SECTIONS: PaperSection[] = [
  { name: 'Section A · Objective questions', marks: 10, questions: 10, detail: 'Ten multiple-choice questions, 1 mark each. Choose the one correct option.' },
  { name: 'Section B · Written responses', marks: 12, questions: 5, detail: 'Five written questions with the marks shown on each. Write your working where needed.' },
]

const CHAPTER_ANSWERS = 'The PDF ends with full answers and marking guidance: every objective answer with a one-line reason, a model answer for each written question, and a mark-by-mark breakdown. Attempt the paper first, then check. Equivalent correct reasoning earns the marks.'

const MIXED_ANSWERS = 'Model answers and marking guidance follow the questions in the PDF, with a mark-by-mark breakdown for every written, extended and case question. Accept equivalent correct reasoning, as the paper itself notes.'

const mixedSections = (): PaperSection[] => [
  { name: 'Section A · Objective questions', marks: 12, questions: 12, detail: 'Twelve multiple-choice questions, 1 mark each.' },
  { name: 'Section B · Short answers', marks: 18, questions: 6, detail: 'Six short-answer questions, 3 marks each.' },
  { name: 'Section C · Extended answers', marks: 20, questions: 4, detail: 'Four extended-answer questions, 5 marks each.' },
  { name: 'Section D · Case questions', marks: 10, questions: 2, detail: 'Two case questions with linked parts, 5 marks each.' },
]

function detail(paperId: string, slug: string, rest: Omit<PaperDetail, 'paperId' | 'slug' | 'path'>): PaperDetail {
  return { paperId, slug, path: `/papers/${slug}/`, ...rest }
}

export const paperDetails: PaperDetail[] = [
  detail('maths-ch01', 'class-8-maths-a-square-and-a-cube-sample-paper', {
    heading: 'Class 8 Maths sample paper: A Square and A Cube',
    seoTitle: 'Class 8 A Square and A Cube Sample Paper with Answers | Evater',
    seoDescription: 'Free Class 8 Maths practice paper on squares and cubes: 15 questions, 22 marks, 45 minutes, with answers and marking guidance. Printable PDF, no sign-in.',
    intro: 'Fifteen original questions on squares, cubes and the patterns hiding inside them, from the first chapter of Ganita Prakash. Download the PDF, work for about 45 minutes, then mark yourself with the included answers.',
    inside: [
      'Squares, cubes and spotting perfect squares and perfect cubes',
      'Units digits of squares and cubes',
      'Square numbers and triangular numbers, including one number that is both',
      'Patterns in sums of odd numbers and in the difference of consecutive squares',
      'Packing unit cubes into a bigger cube, and a first meeting with the 5-12-13 right triangle',
    ],
    sections: CHAPTER_SECTIONS,
    answersNote: CHAPTER_ANSWERS,
    sourceNote: 'Concept reference: NCERT Class 8 Mathematics, Ganita Prakash, Chapter 1: A Square and A Cube. Questions and model answers are original Evater practice material, not an official NCERT paper or marking scheme.',
    studyGuide: { label: 'Read the Squares and Cubes study guide', href: '/blog/class-8-maths-a-square-and-a-cube-study-resources/' },
  }),
  detail('maths-ch02', 'class-8-maths-power-play-sample-paper', {
    heading: 'Class 8 Maths sample paper: Power Play',
    seoTitle: 'Class 8 Power Play Sample Paper with Answers (Exponents) | Evater',
    seoDescription: 'Free Class 8 Maths practice paper on exponents and scientific notation: 15 questions, 22 marks, 45 minutes, with answers and marking guidance. Printable PDF.',
    intro: 'Fifteen original questions on powers, exponent laws and scientific notation. Download the PDF, give yourself about 45 minutes, then check every answer with the marking guidance at the end.',
    inside: [
      'Laws of exponents for multiplying, dividing and raising powers to powers',
      'Zero and negative exponents, and why any non-zero number to the power 0 is 1',
      'Scientific (standard) notation for large and small numbers',
      'Comparing the size of powers, such as 2^10 against 10^2',
      'Repeated doubling: paper folding and a doubling-bacteria problem',
    ],
    sections: CHAPTER_SECTIONS,
    answersNote: CHAPTER_ANSWERS,
    sourceNote: 'Concept reference: NCERT Class 8 Mathematics, Ganita Prakash, Chapter 2: Power Play. Questions and model answers are original Evater practice material, not an official NCERT paper or marking scheme.',
    studyGuide: { label: 'Read the Power Play study guide', href: '/blog/class-8-maths-power-play-study-resources/' },
  }),
  detail('maths-ch03', 'class-8-maths-a-story-of-numbers-sample-paper', {
    heading: 'Class 8 Maths sample paper: A Story of Numbers',
    seoTitle: 'Class 8 A Story of Numbers Sample Paper with Answers | Evater',
    seoDescription: 'Free Class 8 Maths practice paper on number systems: Roman numerals, binary, base 5 and place value. 15 questions with answers and marking guidance.',
    intro: 'Fifteen original questions travelling through Roman, Egyptian, Babylonian, binary and Hindu-Arabic numbers. Download the PDF, work for about 45 minutes, then mark yourself with the included answers.',
    inside: [
      'Reading and writing Roman numerals, including subtractive pairs',
      'The Indian place-value system and reading large numbers',
      'Converting between base 10, binary (base 2) and base 5',
      'The bases behind the ancient Egyptian and Babylonian systems',
      'Why the Hindu-Arabic system handles large numbers well, and why computers use binary',
    ],
    sections: CHAPTER_SECTIONS,
    answersNote: CHAPTER_ANSWERS,
    sourceNote: 'Concept reference: NCERT Class 8 Mathematics, Ganita Prakash, Chapter 3: A Story of Numbers. Questions and model answers are original Evater practice material, not an official NCERT paper or marking scheme.',
    studyGuide: { label: 'Read the Story of Numbers study guide', href: '/blog/class-8-maths-a-story-of-numbers-study-resources/' },
  }),
  detail('maths-ch04', 'class-8-maths-quadrilaterals-sample-paper', {
    heading: 'Class 8 Maths sample paper: Quadrilaterals',
    seoTitle: 'Class 8 Quadrilaterals Sample Paper with Answers (PDF) | Evater',
    seoDescription: 'Free Class 8 Maths practice paper on quadrilaterals: parallelograms, rectangles, rhombuses, kites and diagonals. 15 questions with answers, printable PDF.',
    intro: 'Fifteen original questions on four-sided shapes: angle sums, diagonals and the special families of parallelograms. Download the PDF, work for about 45 minutes, then mark yourself with the included answers.',
    inside: [
      'The 360-degree angle sum of any quadrilateral and finding missing angles',
      'Properties of parallelograms: opposite sides and opposite angles',
      'Diagonals of rectangles, rhombuses and squares, and what makes each special',
      'Kites and the co-interior angles between parallel sides',
      'Angles in a given ratio, and finding a rhombus side from its diagonals',
    ],
    sections: CHAPTER_SECTIONS,
    answersNote: CHAPTER_ANSWERS,
    sourceNote: 'Concept reference: NCERT Class 8 Mathematics, Ganita Prakash, Chapter 4: Quadrilaterals. Questions and model answers are original Evater practice material, not an official NCERT paper or marking scheme.',
    studyGuide: { label: 'Read the Quadrilaterals study guide', href: '/blog/class-8-maths-quadrilaterals-study-resources/' },
  }),
  detail('maths-ch05', 'class-8-maths-number-play-sample-paper', {
    heading: 'Class 8 Maths sample paper: Number Play',
    seoTitle: 'Class 8 Number Play Sample Paper with Answers (PDF) | Evater',
    seoDescription: 'Free Class 8 Maths practice paper on divisibility, primes, HCF and LCM: 15 questions, 22 marks, 45 minutes, with answers and marking guidance. Printable PDF.',
    intro: 'Fifteen original questions on divisibility rules, primes, factors and multiples. Download the PDF, work for about 45 minutes, then mark yourself with the included answers.',
    inside: [
      'Divisibility rules for 3, 4, 5, 9 and 11, with multi-digit tests',
      'Prime numbers and prime factorisation',
      'HCF by listing factors and LCM in a bells-ringing-together problem',
      'Why every even number greater than 2 cannot be prime',
      'What follows when a number is divisible by both 2 and 3',
    ],
    sections: CHAPTER_SECTIONS,
    answersNote: CHAPTER_ANSWERS,
    sourceNote: 'Concept reference: NCERT Class 8 Mathematics, Ganita Prakash, Chapter 5: Number Play. Questions and model answers are original Evater practice material, not an official NCERT paper or marking scheme.',
    studyGuide: { label: 'Read the Number Play study guide', href: '/blog/class-8-maths-number-play-study-resources/' },
  }),
  detail('maths-ch06', 'class-8-maths-we-distribute-yet-things-multiply-sample-paper', {
    heading: 'Class 8 Maths sample paper: We Distribute, Yet Things Multiply',
    seoTitle: 'Class 8 Algebraic Identities Sample Paper with Answers | Evater',
    seoDescription: 'Free Class 8 Maths practice paper on the distributive property and identities: 15 questions, 22 marks, 45 minutes, with answers and marking guidance. Printable PDF.',
    intro: 'Fifteen original questions on expanding brackets and the three great identities. Download the PDF, work for about 45 minutes, then mark yourself with the included answers.',
    inside: [
      'The distributive property and expanding expressions like 3(x + 5) and (x + 2)(x + 5)',
      'The identities (a + b)^2, (a - b)^2 and a^2 - b^2',
      'Fast multiplication with identities: 13^2, 99^2, 102 x 98 and 53 x 47',
      'Spotting the classic mistake in (x + 3)^2',
      'Verifying identities by substitution and by splitting a rectangle into parts',
    ],
    sections: CHAPTER_SECTIONS,
    answersNote: CHAPTER_ANSWERS,
    sourceNote: 'Concept reference: NCERT Class 8 Mathematics, Ganita Prakash, Chapter 6: We Distribute, Yet Things Multiply. Questions and model answers are original Evater practice material, not an official NCERT paper or marking scheme.',
    studyGuide: { label: 'Read the algebraic identities study guide', href: '/blog/class-8-maths-we-distribute-yet-things-multiply-study-resources/' },
  }),
  detail('maths-ch07', 'class-8-maths-proportional-reasoning-1-sample-paper', {
    heading: 'Class 8 Maths sample paper: Proportional Reasoning-1',
    seoTitle: 'Class 8 Ratio and Proportion Sample Paper with Answers | Evater',
    seoDescription: 'Free Class 8 Maths practice paper on ratios and direct proportion: 15 questions, 22 marks, 45 minutes, with answers and marking guidance. Printable PDF.',
    intro: 'Fifteen original questions on ratios, the unitary method and direct proportion in everyday settings. Download the PDF, work for about 45 minutes, then mark yourself with the included answers.',
    inside: [
      'Writing ratios in simplest form, including ratios with mixed units',
      'The unitary method for prices, recipes and quantities',
      'Sharing an amount in a given ratio',
      'Direct proportion: travel at constant speed, map scales and shadow lengths',
      'Enlarging a photo without distortion',
    ],
    sections: CHAPTER_SECTIONS,
    answersNote: CHAPTER_ANSWERS,
    sourceNote: 'Concept reference: NCERT Class 8 Mathematics, Ganita Prakash, Chapter 7: Proportional Reasoning-1. Questions and model answers are original Evater practice material, not an official NCERT paper or marking scheme.',
    studyGuide: { label: 'Read the Proportional Reasoning-1 study guide', href: '/blog/class-8-maths-proportional-reasoning-1-study-resources/' },
  }),
  detail('maths-ch08', 'class-8-maths-fractions-in-disguise-sample-paper', {
    heading: 'Class 8 Maths sample paper: Fractions in Disguise',
    seoTitle: 'Class 8 Percentages Sample Paper with Answers (PDF) | Evater',
    seoDescription: 'Free Class 8 Maths practice paper on percentages, discounts, profit and interest: 15 questions, 22 marks, 45 minutes, with answers and marking guidance.',
    intro: 'Fifteen original questions on percentages and the fractions hiding inside them: discounts, profit, scores and simple interest-style growth. Download the PDF, work for about 45 minutes, then mark yourself with the included answers.',
    inside: [
      'Converting between fractions and percentages, and finding a percentage of a quantity',
      'Percentage increase and decrease, including a successive-change trap',
      'Discounts and selling prices',
      'Profit, profit percentage and a shopkeeper’s costs',
      'Comparing test scores fairly with percentages, and simple yearly growth on a deposit',
    ],
    sections: CHAPTER_SECTIONS,
    answersNote: CHAPTER_ANSWERS,
    sourceNote: 'Concept reference: NCERT Class 8 Mathematics, Ganita Prakash, Chapter 8: Fractions in Disguise. Questions and model answers are original Evater practice material, not an official NCERT paper or marking scheme.',
    studyGuide: { label: 'Read the percentages study guide', href: '/blog/class-8-maths-fractions-in-disguise-study-resources/' },
  }),
  detail('pythagoras', 'class-8-maths-pythagoras-sample-paper', {
    heading: 'Class 8 Maths sample paper: The Baudhayana-Pythagoras theorem',
    seoTitle: 'Class 8 Pythagoras Theorem Sample Paper with Answers | Evater',
    seoDescription: 'Free Class 8 Maths practice paper on the Pythagoras theorem: hypotenuse, diagonals and right-triangle checks. 15 questions with answers and marking guidance.',
    intro: 'Fifteen original questions on the Baudhayana-Pythagoras theorem: computing sides, testing triples and using the theorem inside squares, rectangles and ladders. Download the PDF, work for about 45 minutes, then mark yourself with the included answers.',
    inside: [
      'Finding the hypotenuse and unknown legs of right triangles',
      'Testing whether given side lengths can form a right triangle',
      'Diagonals of squares and rectangles, with exact and decimal answers',
      'Word problems such as a ladder against a wall',
      'Pythagorean triples, building integer rectangles and the area of an equilateral triangle',
    ],
    sections: CHAPTER_SECTIONS,
    answersNote: CHAPTER_ANSWERS,
    sourceNote: 'Concept reference: the Pythagorean theorem; all numerical examples in the paper were independently calculated. Questions and model answers are original Evater practice material, not an official NCERT paper or marking scheme.',
    studyGuide: { label: 'Read the Baudhayana-Pythagoras study guide', href: '/blog/ncert-class-8-maths-baudhayana-pythagoras-theorem/' },
  }),
  detail('maths-ch10', 'class-8-maths-proportional-reasoning-2-sample-paper', {
    heading: 'Class 8 Maths sample paper: Proportional Reasoning-2',
    seoTitle: 'Class 8 Inverse Proportion and Pie Charts Sample Paper | Evater',
    seoDescription: 'Free Class 8 Maths practice paper on inverse proportion, map scales and pie charts: 15 questions, 22 marks, 45 minutes, with answers and marking guidance.',
    intro: 'Fifteen original questions taking proportional reasoning further: bigger map scales, pie charts and inverse proportion. Download the PDF, work for about 45 minutes, then mark yourself with the included answers.',
    inside: [
      'Map scales such as 1 : 1,00,000 and 1 : 2,50,000',
      'Sharing in two-part and three-part ratios, and mixing in a given ratio',
      'Pie chart angles for class data, and why the angles sum to 360 degrees',
      'Inverse proportion: workers and days, speed and time, pipes filling a tank',
      'Telling direct and inverse proportion apart in everyday situations',
    ],
    sections: CHAPTER_SECTIONS,
    answersNote: CHAPTER_ANSWERS,
    sourceNote: 'Concept reference: NCERT Class 8 Mathematics, Ganita Prakash, Chapter 10: Proportional Reasoning-2. Questions and model answers are original Evater practice material, not an official NCERT paper or marking scheme.',
    studyGuide: { label: 'Read the Proportional Reasoning-2 study guide', href: '/blog/class-8-maths-proportional-reasoning-2-study-resources/' },
  }),
  detail('maths-ch11', 'class-8-maths-exploring-some-geometric-themes-sample-paper', {
    heading: 'Class 8 Maths sample paper: Exploring Some Geometric Themes',
    seoTitle: 'Class 8 Fractals and Solids Sample Paper with Answers | Evater',
    seoDescription: 'Free Class 8 Maths practice paper on fractals, nets and polyhedra: 15 questions, 22 marks, 45 minutes, with answers and marking guidance. Printable PDF.',
    intro: 'Fifteen original questions on fractals, nets and counting the parts of solids. Download the PDF, work for about 45 minutes, then mark yourself with the included answers.',
    inside: [
      'Self-similarity and the construction rules of the Sierpinski gasket and carpet',
      'The starting shape of the Koch snowflake',
      'Counting faces, edges and vertices of cubes, prisms and pyramids',
      'Checking Euler’s rule, faces + vertices - edges = 2',
      'Nets of cubes and other solids, and an area computation on a Sierpinski carpet',
    ],
    sections: CHAPTER_SECTIONS,
    answersNote: CHAPTER_ANSWERS,
    sourceNote: 'Concept reference: NCERT Class 8 Mathematics, Ganita Prakash, Chapter 11: Exploring Some Geometric Themes. Questions and model answers are original Evater practice material, not an official NCERT paper or marking scheme.',
    studyGuide: { label: 'Read the Geometric Themes study guide', href: '/blog/class-8-maths-exploring-some-geometric-themes-study-resources/' },
  }),
  detail('maths-ch12', 'class-8-maths-tales-by-dots-and-lines-sample-paper', {
    heading: 'Class 8 Maths sample paper: Tales by Dots and Lines',
    seoTitle: 'Class 8 Mean and Graphs Sample Paper with Answers | Evater',
    seoDescription: 'Free Class 8 Maths practice paper on the mean and data graphs: 15 questions, 22 marks, 45 minutes, with answers and marking guidance. Printable PDF.',
    intro: 'Fifteen original questions on the mean and on choosing the right graph for data. Download the PDF, work for about 45 minutes, then mark yourself with the included answers.',
    inside: [
      'Finding the mean of a data set and the total behind a known mean',
      'How the mean shifts when a value is added or removed',
      'The fair-share meaning of the mean and balancing distances above and below it',
      'When a line graph beats a bar graph, such as temperature through a day',
      'A cricket-scores problem that combines the mean with a new innings',
    ],
    sections: CHAPTER_SECTIONS,
    answersNote: CHAPTER_ANSWERS,
    sourceNote: 'Concept reference: NCERT Class 8 Mathematics, Ganita Prakash, Chapter 12: Tales by Dots and Lines. Questions and model answers are original Evater practice material, not an official NCERT paper or marking scheme.',
    studyGuide: { label: 'Read the Tales by Dots and Lines study guide', href: '/blog/class-8-maths-tales-by-dots-and-lines-study-resources/' },
  }),
  detail('maths-ch13', 'class-8-maths-algebra-play-sample-paper', {
    heading: 'Class 8 Maths sample paper: Algebra Play',
    seoTitle: 'Class 8 Algebra Puzzles Sample Paper with Answers | Evater',
    seoDescription: 'Free Class 8 Maths practice paper on number tricks, pyramids and simple equations: 15 questions, 22 marks, 45 minutes, with answers and marking guidance.',
    intro: 'Fifteen original questions turning algebra into puzzles: number tricks, sum pyramids and calendar patterns. Download the PDF, work for about 45 minutes, then mark yourself with the included answers.',
    inside: [
      'Number tricks and why they always land on the same answer',
      'Sum pyramids where each brick is the sum of the two below it',
      'Calendar 2 x 2 grid patterns and finding the four dates from their total',
      'Divisibility by 3 and 9 with digit sums, and why the test works',
      'Simple equations, and splitting a number to maximise the product of its parts',
    ],
    sections: CHAPTER_SECTIONS,
    answersNote: CHAPTER_ANSWERS,
    sourceNote: 'Concept reference: NCERT Class 8 Mathematics, Ganita Prakash, Chapter 13: Algebra Play. Questions and model answers are original Evater practice material, not an official NCERT paper or marking scheme.',
    studyGuide: { label: 'Read the Algebra Play study guide', href: '/blog/class-8-maths-algebra-play-study-resources/' },
  }),
  detail('maths-ch14', 'class-8-maths-area-sample-paper', {
    heading: 'Class 8 Maths sample paper: Area',
    seoTitle: 'Class 8 Area Sample Paper with Answers (PDF) | Evater',
    seoDescription: 'Free Class 8 Maths practice paper on areas of triangles, parallelograms, rhombuses and trapeziums: 15 questions with answers and marking guidance.',
    intro: 'Fifteen original questions on area: triangles, parallelograms, rhombuses and trapeziums, plus the reasoning that connects them. Download the PDF, work for about 45 minutes, then mark yourself with the included answers.',
    inside: [
      'Area of a triangle from its base and perpendicular height',
      'Areas of parallelograms, rhombuses (from diagonals) and trapeziums',
      'Triangles on the same base between the same pair of parallel lines',
      'How a median or a diagonal splits a shape into equal areas',
      'Working backwards from a known area to a missing base or height',
    ],
    sections: CHAPTER_SECTIONS,
    answersNote: CHAPTER_ANSWERS,
    sourceNote: 'Concept reference: NCERT Class 8 Mathematics, Ganita Prakash, Chapter 14: Area. Questions and model answers are original Evater practice material, not an official NCERT paper or marking scheme.',
    studyGuide: { label: 'Read the Area study guide', href: '/blog/ncert-class-8-maths-area-study-guide/' },
  }),
  detail('science-ch01', 'class-8-science-exploring-the-investigative-world-of-science-sample-paper', {
    heading: 'Class 8 Science sample paper: Exploring the Investigative World of Science',
    seoTitle: 'Class 8 Scientific Investigation Sample Paper with Answers | Evater',
    seoDescription: 'Free Class 8 Science practice paper on fair tests and investigation skills: 15 questions, 22 marks, 45 minutes, with answers and marking guidance. Printable PDF.',
    intro: 'Fifteen original questions on how science actually works: testable questions, fair tests and honest conclusions. Download the PDF, work for about 45 minutes, then mark yourself with the included answers.',
    inside: [
      'How investigations begin, and what makes a question testable',
      'Changing one variable while keeping the others the same',
      'Recording everything you observe, including measurements and surprises',
      'Repeating trials and handling results that disagree with your prediction',
      'Features of Earth that make life possible, and why science is not limited to laboratories',
    ],
    sections: CHAPTER_SECTIONS,
    answersNote: CHAPTER_ANSWERS,
    sourceNote: 'Concept reference: NCERT Class 8 Science, Curiosity, Chapter 1: Exploring the Investigative World of Science. Questions and model answers are original Evater practice material, not an official NCERT paper or marking scheme.',
    studyGuide: { label: 'Read the investigation study guide', href: '/blog/ncert-class-8-science-investigative-world-study-guide/' },
  }),
  detail('science-ch02', 'class-8-science-the-invisible-living-world-beyond-our-naked-eye-sample-paper', {
    heading: 'Class 8 Science sample paper: The Invisible Living World',
    seoTitle: 'Class 8 Microorganisms Sample Paper with Answers (PDF) | Evater',
    seoDescription: 'Free Class 8 Science practice paper on cells and microorganisms: 15 questions, 22 marks, 45 minutes, with answers and marking guidance. Printable PDF.',
    intro: 'Fifteen original questions on cells and the microorganisms we cannot see. Download the PDF, work for about 45 minutes, then mark yourself with the included answers.',
    inside: [
      'Hooke’s first cells and the first observations of living microorganisms',
      'Plant vs animal cells, and where a bacterium keeps its genetic material',
      'Viruses and the foldable paper microscope',
      'Fermentation in idli batter and helpful root-nodule bacteria in legumes',
      'Food spoilage, refrigeration and using antibiotics responsibly',
    ],
    sections: CHAPTER_SECTIONS,
    answersNote: CHAPTER_ANSWERS,
    sourceNote: 'Concept reference: NCERT Class 8 Science, Curiosity, Chapter 2: The Invisible Living World: Beyond Our Naked Eye. Questions and model answers are original Evater practice material, not an official NCERT paper or marking scheme.',
    studyGuide: { label: 'Read the microorganisms study guide', href: '/blog/class-8-science-the-invisible-living-world-beyond-our-naked-eye-study-resources/' },
  }),
  detail('science-ch03', 'class-8-science-health-the-ultimate-treasure-sample-paper', {
    heading: 'Class 8 Science sample paper: Health: The Ultimate Treasure',
    seoTitle: 'Class 8 Health and Disease Sample Paper with Answers | Evater',
    seoDescription: 'Free Class 8 Science practice paper on health, diseases, immunity and hygiene: 15 questions, 22 marks, 45 minutes, with answers and marking guidance.',
    intro: 'Fifteen original questions on staying healthy: how diseases spread, how immunity works and which habits protect you. Download the PDF, work for about 45 minutes, then mark yourself with the included answers.',
    inside: [
      'What being healthy means, and communicable vs non-communicable diseases',
      'How ringworm, chikungunya and water-borne diseases reach people',
      'Vaccination, immunity after chickenpox and why antibiotics do not fix viral fevers',
      'Hygiene for food handlers and preventing mosquito breeding',
      'Lifestyle choices: sleep, breakfast, screen time and handling exam stress',
    ],
    sections: CHAPTER_SECTIONS,
    answersNote: CHAPTER_ANSWERS,
    sourceNote: 'Concept reference: NCERT Class 8 Science, Curiosity, Chapter 3: Health: The Ultimate Treasure. Questions and model answers are original Evater practice material, not an official NCERT paper or marking scheme.',
    studyGuide: { label: 'Read the health study guide', href: '/blog/ncert-class-8-science-health-ultimate-treasure-study-guide/' },
  }),
  detail('electricity', 'class-8-science-electricity-sample-paper', {
    heading: 'Class 8 Science sample paper: Electricity: Magnetic and Heating Effects',
    seoTitle: 'Class 8 Electricity Sample Paper with Answers (PDF) | Evater',
    seoDescription: 'Free Class 8 Science practice paper on the magnetic and heating effects of current and electromagnets: 15 questions with answers and marking guidance.',
    intro: 'Fifteen original questions on what electric current does: deflecting compasses, building electromagnets and heating wires. Download the PDF, work for about 45 minutes, then mark yourself with the included answers.',
    inside: [
      'The magnetic effect of current: a compass needle deflecting near a wire',
      'Making a simple electromagnet from a cell, insulated wire and an iron nail',
      'What strengthens or weakens an electromagnet',
      'The heating effect of current, and why a wire can stay warm after switch-off',
      'Safe ways to detect current without touching, and electric vs smoky indoor cooking',
    ],
    sections: CHAPTER_SECTIONS,
    answersNote: CHAPTER_ANSWERS,
    sourceNote: 'Concept references: current and magnetic effects; the point-of-use smoke distinction. Questions and model answers are original Evater practice material, not an official NCERT paper or marking scheme.',
    studyGuide: { label: 'Read the electricity study guide', href: '/blog/ncert-class-8-science-electricity-magnetic-heating-effects-study-guide/' },
  }),
  detail('science-ch05', 'class-8-science-exploring-forces-sample-paper', {
    heading: 'Class 8 Science sample paper: Exploring Forces',
    seoTitle: 'Class 8 Forces Sample Paper with Answers (PDF) | Evater',
    seoDescription: 'Free Class 8 Science practice paper on forces, friction, gravity and buoyancy: 15 questions, 22 marks, 45 minutes, with answers and marking guidance.',
    intro: 'Fifteen original questions on pushes and pulls: contact and non-contact forces, friction, weight and buoyancy. Download the PDF, work for about 45 minutes, then mark yourself with the included answers.',
    inside: [
      'Contact vs non-contact forces: muscular, magnetic, electrostatic and gravitational',
      'The effects a force can produce: starting motion and changing shape',
      'Friction slowing bicycles and swings, and why powder is sprinkled on a carrom board',
      'Mass vs weight, including what changes on the Moon',
      'Buoyancy: the upward force of water on a floating boat',
    ],
    sections: CHAPTER_SECTIONS,
    answersNote: CHAPTER_ANSWERS,
    sourceNote: 'Concept reference: NCERT Class 8 Science, Curiosity, Chapter 5: Exploring Forces. Questions and model answers are original Evater practice material, not an official NCERT paper or marking scheme.',
    studyGuide: { label: 'Read the forces study guide', href: '/blog/class-8-science-exploring-forces-study-resources/' },
  }),
  detail('science-ch06', 'class-8-science-pressure-winds-storms-and-cyclones-sample-paper', {
    heading: 'Class 8 Science sample paper: Pressure, Winds, Storms, and Cyclones',
    seoTitle: 'Class 8 Pressure and Winds Sample Paper with Answers | Evater',
    seoDescription: 'Free Class 8 Science practice paper on pressure, atmospheric pressure, storms and cyclones: 15 questions with answers and marking guidance. Printable PDF.',
    intro: 'Fifteen original questions on pressure in daily life and the weather it drives. Download the PDF, work for about 45 minutes, then mark yourself with the included answers.',
    inside: [
      'Pressure, its SI unit, and why bag straps and dam walls are shaped the way they are',
      'Pressure in liquids and gases: the crumpling juice carton and drinking through a straw',
      'Land and sea breezes from unequal heating of the Earth',
      'How thunderstorms form and how a lightning conductor protects a building',
      'What people should do when a cyclone warning arrives',
    ],
    sections: CHAPTER_SECTIONS,
    answersNote: CHAPTER_ANSWERS,
    sourceNote: 'Concept reference: NCERT Class 8 Science, Curiosity, Chapter 6: Pressure, Winds, Storms, and Cyclones. Questions and model answers are original Evater practice material, not an official NCERT paper or marking scheme.',
    studyGuide: { label: 'Read the pressure and winds study guide', href: '/blog/ncert-class-8-science-pressure-winds-storms-cyclones-study-guide/' },
  }),
  detail('science-ch07', 'class-8-science-particulate-nature-of-matter-sample-paper', {
    heading: 'Class 8 Science sample paper: Particulate Nature of Matter',
    seoTitle: 'Class 8 Particle Model of Matter Sample Paper | Evater',
    seoDescription: 'Free Class 8 Science practice paper on the particle model, states of matter and diffusion: 15 questions, 22 marks, 45 minutes, with answers and marking guidance.',
    intro: 'Fifteen original questions on the particle model: why solids keep their shape, smells travel and puddles disappear. Download the PDF, work for about 45 minutes, then mark yourself with the included answers.',
    inside: [
      'The main ideas of the particle model of matter',
      'Solids, liquids and gases: shape, volume and compressing a sealed syringe',
      'Diffusion: an agarbatti smell crossing a room and salt seasoning a whole pot',
      'Why warmth speeds up dissolving and drying',
      'Melting, evaporation and where rainwater goes when a pavement dries',
    ],
    sections: CHAPTER_SECTIONS,
    answersNote: CHAPTER_ANSWERS,
    sourceNote: 'Concept reference: NCERT Class 8 Science, Curiosity, Chapter 7: Particulate Nature of Matter. Questions and model answers are original Evater practice material, not an official NCERT paper or marking scheme.',
    studyGuide: { label: 'Read the particle model study guide', href: '/blog/ncert-class-8-science-particulate-nature-of-matter-study-guide/' },
  }),
  detail('science-ch08', 'class-8-science-nature-of-matter-elements-compounds-and-mixtures-sample-paper', {
    heading: 'Class 8 Science sample paper: Nature of Matter: Elements, Compounds, and Mixtures',
    seoTitle: 'Class 8 Elements Compounds Mixtures Sample Paper | Evater',
    seoDescription: 'Free Class 8 Science practice paper on elements, compounds and mixtures: 15 questions, 22 marks, 45 minutes, with answers and marking guidance. Printable PDF.',
    intro: 'Fifteen original questions on sorting the material world into elements, compounds and mixtures. Download the PDF, work for about 45 minutes, then mark yourself with the included answers.',
    inside: [
      'Telling elements, compounds and mixtures apart, with everyday examples',
      'Fixed ratios in compounds such as water',
      'Separating mixtures with a magnet, dissolving and filtering',
      'Alloys such as brass, and why they count as mixtures',
      'Iron and sulfur: a separable mixture vs the compound formed on heating',
    ],
    sections: CHAPTER_SECTIONS,
    answersNote: CHAPTER_ANSWERS,
    sourceNote: 'Concept reference: NCERT Class 8 Science, Curiosity, Chapter 8: Nature of Matter: Elements, Compounds, and Mixtures. Questions and model answers are original Evater practice material, not an official NCERT paper or marking scheme.',
    studyGuide: { label: 'Read the elements and mixtures study guide', href: '/blog/ncert-class-8-science-elements-compounds-mixtures-study-guide/' },
  }),
  detail('science-ch09', 'class-8-science-the-amazing-world-of-solutes-solvents-and-solutions-sample-paper', {
    heading: 'Class 8 Science sample paper: Solutes, Solvents, and Solutions',
    seoTitle: 'Class 8 Solutions and Density Sample Paper with Answers | Evater',
    seoDescription: 'Free Class 8 Science practice paper on solutions, solubility and density: 15 questions, 22 marks, 45 minutes, with answers and marking guidance. Printable PDF.',
    intro: 'Fifteen original questions on what dissolves, how fast, and why some things float. Download the PDF, work for about 45 minutes, then mark yourself with the included answers.',
    inside: [
      'Solutes, solvents and what counts as a true solution',
      'Saturated solutions and how heating lets more solute dissolve',
      'Stirring and temperature: comparing what changes dissolving speed',
      'Density from mass and volume, with a laboratory-style calculation',
      'Floating and sinking, and how heating water changes its density',
    ],
    sections: CHAPTER_SECTIONS,
    answersNote: CHAPTER_ANSWERS,
    sourceNote: 'Concept reference: NCERT Class 8 Science, Curiosity, Chapter 9: The Amazing World of Solutes, Solvents, and Solutions. Questions and model answers are original Evater practice material, not an official NCERT paper or marking scheme.',
    studyGuide: { label: 'Read the solutions and density study guide', href: '/blog/ncert-class-8-science-solutes-solvents-solutions-density-study-guide/' },
  }),
  detail('light', 'class-8-science-light-sample-paper', {
    heading: 'Class 8 Science sample paper: Light: Mirrors and Lenses',
    seoTitle: 'Class 8 Light Sample Paper: Mirrors and Lenses | Evater',
    seoDescription: 'Free Class 8 Science practice paper on reflection, mirrors, lenses and refraction: 15 questions, 22 marks, 45 minutes, with answers and marking guidance.',
    intro: 'Fifteen original questions on reflection, mirrors, lenses and refraction. Download the PDF, work for about 45 minutes, then mark yourself with the included answers.',
    inside: [
      'Reflection from polished surfaces and measuring angles from the normal',
      'Plane mirrors: where the image forms and how far behind the mirror it sits',
      'Convex mirrors for wide road-corner views and concave mirrors for close-up magnification',
      'Lenses: the hand magnifier and how a concave lens spreads a parallel beam',
      'Refraction from air into water, and why a lit page is visible from many positions',
    ],
    sections: CHAPTER_SECTIONS,
    answersNote: CHAPTER_ANSWERS,
    sourceNote: 'Concept references: OpenStax Physics, Reflection and Lenses. Questions and model answers are original Evater practice material, not an official NCERT paper or marking scheme.',
    studyGuide: { label: 'Read the light study guide', href: '/blog/ncert-class-8-science-light-mirrors-lenses-study-guide/' },
  }),
  detail('science-ch11', 'class-8-science-keeping-time-with-the-skies-sample-paper', {
    heading: 'Class 8 Science sample paper: Keeping Time with the Skies',
    seoTitle: 'Class 8 Moon Phases and Calendars Sample Paper | Evater',
    seoDescription: 'Free Class 8 Science practice paper on the Moon, its phases and calendars: 15 questions, 22 marks, 45 minutes, with answers and marking guidance. Printable PDF.',
    intro: 'Fifteen original questions on the Moon’s phases and how the skies gave us days, months and years. Download the PDF, work for about 45 minutes, then mark yourself with the included answers.',
    inside: [
      'Days, lunar months and years from the repeating motions of Earth and the Moon',
      'Why the Moon shows phases, and why Earth’s shadow is not the cause',
      'Leap years, luni-solar calendars and intercalary months',
      'Reading moonrise clues: which phase is highest around midnight',
      'Designing and checking calendar models with arithmetic',
    ],
    sections: CHAPTER_SECTIONS,
    answersNote: CHAPTER_ANSWERS,
    sourceNote: 'Concept reference: NCERT Class 8 Science, Curiosity, Chapter 11: Keeping Time with the Skies. Questions and model answers are original Evater practice material, not an official NCERT paper or marking scheme.',
    studyGuide: { label: 'Read the skies and calendars study guide', href: '/blog/ncert-class-8-science-keeping-time-skies-study-guide/' },
  }),
  detail('science-ch12', 'class-8-science-how-nature-works-in-harmony-sample-paper', {
    heading: 'Class 8 Science sample paper: How Nature Works in Harmony',
    seoTitle: 'Class 8 Ecosystems Sample Paper with Answers (PDF) | Evater',
    seoDescription: 'Free Class 8 Science practice paper on ecosystems, food webs and decomposers: 15 questions, 22 marks, 45 minutes, with answers and marking guidance.',
    intro: 'Fifteen original questions on ecosystems: who eats whom, who breaks things down, and how people fit in. Download the PDF, work for about 45 minutes, then mark yourself with the included answers.',
    inside: [
      'Populations, communities and ecosystems, including non-living components',
      'Producers and the recycling work of decomposers',
      'Food chains and food webs, with a bird that eats two kinds of prey',
      'Mangroves as storm protection and fish nurseries, and human-made ecosystems',
      'How garden choices ripple through decomposers, pollinators and plants',
    ],
    sections: CHAPTER_SECTIONS,
    answersNote: CHAPTER_ANSWERS,
    sourceNote: 'Concept reference: NCERT Class 8 Science, Curiosity, Chapter 12: How Nature Works in Harmony. Questions and model answers are original Evater practice material, not an official NCERT paper or marking scheme.',
    studyGuide: { label: 'Read the ecosystems study guide', href: '/blog/class-8-science-how-nature-works-in-harmony-study-resources/' },
  }),
  detail('science-ch13', 'class-8-science-our-home-earth-a-unique-life-sustaining-planet-sample-paper', {
    heading: 'Class 8 Science sample paper: Our Home: Earth, a Unique Life Sustaining Planet',
    seoTitle: 'Class 8 Earth and Life Sample Paper with Answers | Evater',
    seoDescription: 'Free Class 8 Science practice paper on Earth’s life-supporting systems and biodiversity: 15 questions, 22 marks, 45 minutes, with answers and marking guidance.',
    intro: 'Fifteen original questions on what makes Earth able to sustain life, and how living things continue. Download the PDF, work for about 45 minutes, then mark yourself with the included answers.',
    inside: [
      'Why Earth can host life: distance from the Sun, atmosphere, magnetic field and liquid water',
      'The gases of photosynthesis and the roles of the atmosphere and magnetic field',
      'Biodiversity and geodiversity, and what a field survey would count',
      'Reproduction and continuity, including vegetative propagation from cuttings',
      'How forests, wetlands and city plans affect a life-sustaining planet',
    ],
    sections: CHAPTER_SECTIONS,
    answersNote: CHAPTER_ANSWERS,
    sourceNote: 'Concept reference: NCERT Class 8 Science, Curiosity, Chapter 13: Our Home: Earth, a Unique Life Sustaining Planet. Questions and model answers are original Evater practice material, not an official NCERT paper or marking scheme.',
    studyGuide: { label: 'Read the Earth study guide', href: '/blog/ncert-class-8-science-earth-life-sustaining-planet-study-guide/' },
  }),
  detail('maths-mixed', 'class-8-maths-mixed-chapter-practice-paper', {
    heading: 'Class 8 Maths mixed-chapter practice paper',
    seoTitle: 'Class 8 Maths Mixed-Chapter Practice Paper (60 Marks) | Evater',
    seoDescription: 'Free Class 8 Maths mixed practice paper: 24 questions across all 14 chapters, 60 marks, 150 minutes, with model answers and marking guidance. Printable PDF.',
    intro: 'A longer 24-question paper that samples every Class 8 Maths chapter, from squares and cubes to area. Set aside about 150 minutes, attempt all four sections, then mark yourself with the included model answers.',
    inside: [
      'Objective questions on square patterns, perfect cubes, scientific notation and comparing powers',
      'Quadrilateral angle sums and diagonals, distributive expansion and algebraic identities',
      'Map scales, ratio division and ratio comparisons in the objective section',
      'Written questions on number bases, divisibility, successive percentages, a right triangle, inverse proportion and polyhedra',
      'Extended questions on interpreting data, an algebraic number trick, composite area and perimeter, and a consecutive-square model',
      'Two case questions: ticket-desk ratios with a discount, and a ramp built to scale with area reasoning',
    ],
    sections: mixedSections(),
    answersNote: MIXED_ANSWERS,
    sourceNote: 'Original Evater practice covering selected topics across all 14 chapters, not every syllabus skill. Not an official NCERT or CBSE paper, and there is no internal choice.',
    chapterSample: [
      { chapter: 'A Square and A Cube', questions: 3 },
      { chapter: 'Power Play', questions: 2 },
      { chapter: 'A Story of Numbers', questions: 1 },
      { chapter: 'Quadrilaterals', questions: 2 },
      { chapter: 'Number Play', questions: 1 },
      { chapter: 'We Distribute, Yet Things Multiply', questions: 3 },
      { chapter: 'Proportional Reasoning-1', questions: 4 },
      { chapter: 'Fractions in Disguise', questions: 1 },
      { chapter: 'The Baudhayana-Pythagoras Theorem', questions: 2 },
      { chapter: 'Proportional Reasoning-2', questions: 1 },
      { chapter: 'Exploring Some Geometric Themes', questions: 1 },
      { chapter: 'Tales by Dots and Lines', questions: 1 },
      { chapter: 'Algebra Play', questions: 1 },
      { chapter: 'Area', questions: 1 },
    ],
  }),
  detail('science-mixed', 'class-8-science-mixed-chapter-practice-paper', {
    heading: 'Class 8 Science mixed-chapter practice paper',
    seoTitle: 'Class 8 Science Mixed-Chapter Practice Paper (60 Marks) | Evater',
    seoDescription: 'Free Class 8 Science mixed practice paper: 24 questions across all 13 chapters, 60 marks, 150 minutes, with model answers and marking guidance. Printable PDF.',
    intro: 'A longer 24-question paper that samples every Class 8 Science chapter, from investigation skills to Earth as a life-sustaining planet. Set aside about 150 minutes, attempt all four sections, then mark yourself with the included model answers.',
    inside: [
      'Objective questions on public health, deficiency diseases, immunity and electromagnets',
      'Conductors and insulators, fuse safety, gases, particle behaviour and changes of state',
      'Fair investigations, decomposition and the effects of forces in the objective section',
      'Written questions on designing an investigation, microbial activity, pressure and area, classifying substances, a solubility calculation and mirror images',
      'Extended questions on the Moon and calendars, food-web reasoning, Earth systems and comparing particle models',
      'Two case questions: magnetic and friction forces, and density records',
    ],
    sections: mixedSections(),
    answersNote: MIXED_ANSWERS,
    sourceNote: 'Original Evater practice covering selected topics across all 13 chapters, not every syllabus skill. Not an official NCERT or CBSE paper, and there is no internal choice.',
    chapterSample: [
      { chapter: 'Exploring the Investigative World of Science', questions: 2 },
      { chapter: 'The Invisible Living World: Beyond Our Naked Eye', questions: 2 },
      { chapter: 'Health: The Ultimate Treasure', questions: 3 },
      { chapter: 'Electricity: Magnetic and Heating Effects', questions: 3 },
      { chapter: 'Exploring Forces', questions: 2 },
      { chapter: 'Pressure, Winds, Storms, and Cyclones', questions: 1 },
      { chapter: 'Particulate Nature of Matter', questions: 4 },
      { chapter: 'Nature of Matter: Elements, Compounds, and Mixtures', questions: 1 },
      { chapter: 'The Amazing World of Solutes, Solvents, and Solutions', questions: 2 },
      { chapter: 'Light: Mirrors and Lenses', questions: 1 },
      { chapter: 'Keeping Time with the Skies', questions: 1 },
      { chapter: 'How Nature Works in Harmony', questions: 1 },
      { chapter: 'Our Home: Earth, a Unique Life Sustaining Planet', questions: 1 },
    ],
  }),
]

const bySlug = new Map(paperDetails.map(d => [d.slug, d]))
const byPaperId = new Map(paperDetails.map(d => [d.paperId, d]))

export function getPaperDetailBySlug(slug: string | undefined): PaperDetail | undefined {
  return slug ? bySlug.get(slug) : undefined
}

/** Canonical detail-page path for a catalog paper, used for crawlable card links. */
export function paperDetailPath(paperId: string): string | undefined {
  return byPaperId.get(paperId)?.path
}

/** The catalog paper a detail page describes. */
export function paperForDetail(detailPage: PaperDetail): SamplePaper {
  const paper = samplePapers.find(p => p.id === detailPage.paperId)
  if (!paper) throw new Error(`No sample paper matches detail page ${detailPage.slug}`)
  return paper
}

/** Only chapter papers select a chapter. Mixed papers keep the general chooser. */
export function paperPracticeHref(paperId: string): string | undefined {
  const paper = samplePapers.find(p => p.id === paperId)
  if (!paper) return undefined
  if (paper.format === 'mixed') return '/practice'
  const legacy: Record<string, number> = { light: 10, electricity: 4, pythagoras: 9 }
  const match = paper.id.match(/^(maths|science)-ch(\d{2})$/)
  const chapterNumber = match ? Number(match[2]) : legacy[paper.id]
  if (!chapterNumber) return undefined
  return chapterPracticeHref({ subject: paper.subject === 'Mathematics' ? 'mathematics' : 'science', chapterNumber })
}

/** Direct, crawlable chapter-paper links. No matching by fuzzy titles. */
export function paperForStudyGuide(slug: string): PaperDetail | undefined {
  const href = `/blog/${slug}/`
  const exact = paperDetails.find(paper => paper.studyGuide?.href === href)
  if (exact) return exact
  const aliases: Record<string, string> = {
    'ncert-class-8-maths-square-and-cube-study-guide': 'maths-ch01',
    'ncert-class-8-maths-power-play-study-guide': 'maths-ch02',
    'ncert-class-8-science-invisible-living-world-study-guide': 'science-ch02',
    'ncert-class-8-science-exploring-forces-study-guide': 'science-ch05',
    'ncert-class-8-maths-baudhayana-pythagoras-theorem': 'pythagoras',
    'ncert-class-8-science-light-mirrors-lenses-study-guide': 'light',
    'ncert-class-8-science-nature-ecosystems-harmony-study-guide': 'science-ch12',
  }
  return aliases[slug] ? byPaperId.get(aliases[slug]) : undefined
}
