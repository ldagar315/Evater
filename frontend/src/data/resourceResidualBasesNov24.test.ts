import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
const text = readFileSync(new URL('./resourceGuides/maths-3.md', import.meta.url), 'utf8')
const cases = [[3, "In base two, two units of one place regroup as one unit of the next place. Add one to 111\u2082: the ones place gives two ones, so carry one into the twos place; that also gives two twos, then two fours. The result is 1000\u2082. Check in decimal: 111\u2082 = 4 + 2 + 1 = 7, and 1000\u2082 = 8. The extra written digit records regrouping, not eight new objects.", "Write decimal 13 in base two and add one in that base. A learner writes 1102\u2082. Explain the error and check the correct answer in decimal.", "13 = 8 + 4 + 1, so it is 1101\u2082. Adding one makes two ones: regroup them as one two to get 1110\u2082 = 8 + 4 + 2 = 14. The digit 2 is not allowed in a standard binary numeral; it signals a carry that has not been completed."], [4, "In positional base five, the places from right to left are 1, 5 and 25. Thus 104\u2085 = 1 \u00d7 25 + 0 \u00d7 5 + 4 = 29 in decimal. The zero keeps the 1 in the twenty-fives place; deleting it gives 14\u2085 = 9, a different number. Adding one to 104\u2085 gives 110\u2085: five ones regroup as one five. Decode the result as 25 + 5 = 30 to check.", "Compare 31\u2085 and 31\u2081\u2080. Then write decimal 19 in base five and explain why 34\u2085 + 1\u2085 becomes 40\u2085 rather than 35\u2085.", "31\u2085 means 3 \u00d7 5 + 1 = 16 in decimal, whereas 31\u2081\u2080 means thirty-one. Decimal 19 is 3 \u00d7 5 + 4 = 34\u2085. Adding one regroups five ones into one five, giving 40\u2085 = 20 in decimal. A standard base-five numeral uses only digits 0 through 4. The base label changes the place weights, not the amount being represented."]]
describe('residual bases worked reasoning blocks', () => {
  for (const [topic, ...parts] of cases) it('preserves practice and adds distinct reasoning in topic '+topic, () => {
    const block=text.split('## '+topic+'.')[1].split('\n## ')[0]
    for(const marker of ['### Try it yourself','**Check:**','### Work through an example','### Try a different case','**Reasoning:**']) expect(block).toContain(marker)
    for(const part of parts) expect(block).toContain(part)
  })
  it('retains revision and avoids source detours', () => {
    expect(text).toContain('## Your five-minute revision')
    expect(text).not.toContain('supporting evidence')
    expect(text).not.toMatch(/https?:\/\/(?:ncert|edu\.rsc|www\.csunplugged)/)
  })
})
