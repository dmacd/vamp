# TinyWorlds-P Archive v1 Validation Sample Appendix

This appendix makes the archive-only calibration conditions concrete. It covers all 11 validation conditions used to compute the gap on each grid: held-in base, worlds A–E, and controls A–E. Sealed-test indexes were not read.

## Deterministic selection rule

For each validation index, the generator computes the lower-median document token count, orders documents by absolute distance from that median and then by normalized-story SHA-256 and record ID, and selects the first. A matched control is an equal-count mixture of two materially different arms, so the appendix selects the first candidate from each arm under that same order. This yields one base story, one story per world, and two stories per control on each grid without inspecting story semantics. Every displayed story was reconstructed from its persisted text-shard offset and checked against its exact-story SHA-256. The Markdown renderer removes only invisible trailing line whitespace after that byte-level check.

The noun, verb, adjective, and feature recipe is construction evidence. The model received only the exact story text shown in the block quote; it did not receive recipes, bucket numbers, world labels, source labels, or archive coordinates.

## Initial 8×8 calibration

Partition: `beb9e1e38efdf0447b9421b072c4053fdb7b6156c4814edefa170ec40072f084`.

### Condition inventory

| Validation condition | Occurrences | Active tokens | Lower-median length | Samples below |
| --- | ---: | ---: | ---: | ---: |
| `base/validation` | 91,594 | 17,439,078 | 178 | 1 |
| `world/A/validation` | 7,724 | 1,467,428 | 177 | 1 |
| `world/B/validation` | 7,732 | 1,470,264 | 178 | 1 |
| `world/C/validation` | 7,722 | 1,470,297 | 178 | 1 |
| `world/D/validation` | 7,720 | 1,467,438 | 178 | 1 |
| `world/E/validation` | 7,731 | 1,469,783 | 177 | 1 |
| `control/A/validation` | 7,724 | 1,467,428 | 182 | 2 |
| `control/B/validation` | 7,732 | 1,469,910 | 159 | 2 |
| `control/C/validation` | 7,722 | 1,470,297 | 183 | 2 |
| `control/D/validation` | 7,720 | 1,467,438 | 160 | 2 |
| `control/E/validation` | 7,731 | 1,469,783 | 182 | 2 |

### Selected world topology

| World | Physical cell | Eligible groups | Active tokens |
| :---: | :---: | ---: | ---: |
| A | `N6 × V2` | 77,249 | 14,681,992 |
| B | `N7 × V2` | 77,336 | 14,710,908 |
| C | `N7 × V3` | 77,238 | 14,712,597 |
| D | `N6 × V3` | 77,230 | 14,684,269 |
| E | `N2 × V4` | 77,309 | 14,705,141 |

### What the selected buckets contain

Examples are 12 evenly spaced entries from each alphabetically sorted bucket, not hand-picked themes.

| Bucket | Kind | Words | Archive token mass | Deterministic examples |
| :---: | :---: | ---: | ---: | --- |
| `N2` | noun | 134 | 118,799,837 | `anchor`, `case`, `creek`, `fruit`, `infant`, `license`, `nightmare`, `plate`, `shade`, `sunrise`, `truth`, `zoom` |
| `N6` | noun | 133 | 117,984,291 | `apple`, `bubble`, `costume`, `eye`, `gym`, `land`, `nut`, `police`, `salt`, `spirit`, `team`, `work` |
| `N7` | noun | 133 | 117,983,392 | `acorn`, `carrot`, `field`, `gift`, `luggage`, `paint`, `poppy`, `scarf`, `steel`, `tour`, `wagon`, `zip` |
| `V2` | verb | 49 | 117,619,832 | `act`, `bow`, `enter`, `hang`, `joke`, `observe`, `prevent`, `ride`, `sign`, `steal`, `test`, `zip` |
| `V3` | verb | 49 | 117,618,826 | `bite`, `complete`, `encourage`, `hear`, `kneel`, `move`, `print`, `remember`, `rock`, `shoot`, `try`, `zoom` |
| `V4` | verb | 49 | 117,629,664 | `accept`, `clean`, `examine`, `go`, `learn`, `notice`, `promise`, `save`, `sleep`, `start`, `unlock`, `yawn` |

### Held-in base validation

#### `base/validation`

- Recipe: adjective `helpless`, noun `goat`, verb `dare`; features `none`.
- Recipe cell: `N3 × V7`.
- Length: 178 tokens; condition lower median 178 tokens.
- Source: `GPT-4`; archive member `./data01.json` record 48102.
- Record ID: `archive:./data01.json:48102:2387ddb0b58588fc95bde54aa421bbf1640e57ebf53063c0c86f5057762e122a`.
- Exact-story SHA-256: `779c9db51bbd474d0a6136515bf903d2d925b8b37ae513b5486ca65566a1f085`.

>
> Once upon a time, there was a helpless goat called Daisy. She lived in a small field and all she ever did was eat grass and trot around. Daisy was a very happy goat.
> One day, she dared to do something special. She decided to climb up a big tall tree. She went right to the top and it felt so exciting.
> But then she realised she was stuck way up high and she had no way to get back down. She felt so helpless and scared.
> Just then, a kind farmer came by and saw Daisy up in the tree. He kindly asked if she wanted help getting down. Daisy nodded her head, feeling relieved. The farmer laughed and carefully helped her down.
> Daisy thanked him and skipped back to her pasture, feeling brave and proud of herself. She had dared to do something special and made a new friend!

### World A: N6 × V2

This world contains recipes pairing any of 133 nouns in `N6` with any of 49 verbs in `V2`. Its matched control combines an equal-count same-noun-row arm and same-verb-column arm from held-in validation.

#### `world/A/validation`

- Recipe: adjective `gloomy`, noun `touch`, verb `bow`; features `dialogue`.
- Recipe cell: `N6 × V2`.
- Length: 177 tokens; condition lower median 177 tokens.
- Source: `GPT-4`; archive member `./data38.json` record 14693.
- Record ID: `archive:./data38.json:14693:7100836f991e85a0b36bb770ddf99ef920d56a4c00621cbbcb4012658774689b`.
- Exact-story SHA-256: `ed7623c3f367309db6054259a25222d6e61701c23f79d0ed5c787fffca84cf18`.

> One day, a little dog named Bow went for a walk. Bow felt gloomy because he had no friends to play with. He walked through the park, hoping to find someone to play with.
> As Bow walked, he saw a big tree. Under the tree, there was a little girl named Tina. Tina looked sad too. Bow walked up to her and said, "Hi, I am Bow. Why are you sad?" Tina looked at Bow and said, "I am sad because I have no one to play with."
> Bow wagged his tail and said, "I will be your friend, and we can play together!" Tina smiled and gave Bow a gentle touch on his head. They played all day in the park, and they were not gloomy anymore. Bow and Tina became the best of friends, and they were always happy when they were together.

#### `control/A/validation` — same noun row

- Recipe: adjective `fake`, noun `comet`, verb `bake`; features `dialogue`.
- Recipe cell: `N6 × V6`; control arm `same noun row`.
- Length: 182 tokens; condition lower median 182 tokens.
- Source: `GPT-4`; archive member `./data47.json` record 96080.
- Record ID: `archive:./data47.json:96080:783f2bdc221a2840907006daaf3cf90655b42546130b94bc271d242b992e8444`.
- Exact-story SHA-256: `43605fe9151bca98b93c3764e614972a21c26502b558498c58c65bb2eff83e7d`.

> One day, a little girl named Sue was playing in her room. She had a toy oven and loved to bake. She also had a toy comet that she liked to fly around.
> Sue picked up the comet and said, "Let's bake a cake, comet!" She put the comet in the oven and closed the door. She waited for a few minutes and then opened the door. The comet was still the same. Sue was sad. She wanted a real cake, not a fake one.
> Sue's mom heard her and came into the room. "What's wrong, Sue?" she asked. Sue told her mom about the fake cake. Her mom smiled and said, "Let's bake a real cake together in the kitchen." Sue and her mom went to the kitchen and baked a big, yummy cake. Sue was happy and shared the cake with her toy comet.

#### `control/A/validation` — same verb column

- Recipe: adjective `ancient`, noun `folder`, verb `mail`; features `dialogue`.
- Recipe cell: `N4 × V2`; control arm `same verb column`.
- Length: 182 tokens; condition lower median 182 tokens.
- Source: `GPT-4`; archive member `./data38.json` record 64660.
- Record ID: `archive:./data38.json:64660:a6dc9058f931f0b665174dee3f8a67eff31095e33d90eb5dbc250d9d1256cc32`.
- Exact-story SHA-256: `1ba1c0212dd9105ad9e12f54d97d35d5e9fb1cb4f2bc20cf74f4bab0b1428612`.

> Once upon a time, there was a little boy named Tim. He lived in a small house with his mom and dad. One day, Tim found an ancient folder in the attic. It was very old and dusty. Tim showed the folder to his mom and dad. They were very excited.
> Tim's mom said, "We need to mail this folder to grandpa. He will know what to do with it." So, Tim and his mom went to the post office. They put the folder in a big envelope and mailed it to grandpa's house.
> A few days later, grandpa called Tim. He said, "Thank you for the ancient folder, Tim! It has very special pictures of our family. I will keep it safe." Tim felt happy that he could help his grandpa. And from that day on, Tim loved to explore the attic and find more treasures.

### World B: N7 × V2

This world contains recipes pairing any of 133 nouns in `N7` with any of 49 verbs in `V2`. Its matched control combines an equal-count same-noun-row arm and same-verb-column arm from held-in validation.

#### `world/B/validation`

- Recipe: adjective `long`, noun `present`, verb `belong`; features `twist`.
- Recipe cell: `N7 × V2`.
- Length: 178 tokens; condition lower median 178 tokens.
- Source: `GPT-3.5`; archive member `./data23.json` record 24466.
- Record ID: `archive:./data23.json:24466:f2249b9eb9e5f094b1cab90b7d3b8643286b25b07eff29060cf4a1d68e249edc`.
- Exact-story SHA-256: `a649ab822426d262a9e1b7a752896c831f6923d611b45a26f7004aa6f85324ab`.

> Once upon a time, there was a little girl named Lily. Lily had a long, curly hair and loved to play outside. One day, Lily went to the park with her mommy and daddy. They had a picnic and played on the swings.
> Suddenly, Lily saw a present on the ground. She picked it up and looked around, but nobody was there. She wondered who the present belonged to. She opened it and found a toy inside. She was so happy!
> But then, a little boy came running and said, "That's my present! I lost it earlier." Lily felt sad and wanted to give the present back. But the little boy said, "Don't worry, you can keep it. I have another one at home." Lily was surprised and grateful. She never expected to get a present, and now she had one to keep forever.

#### `control/B/validation` — same noun row

- Recipe: adjective `dizzy`, noun `pin`, verb `sing`; features `none`.
- Recipe cell: `N7 × V6`; control arm `same noun row`.
- Length: 159 tokens; condition lower median 159 tokens.
- Source: `GPT-4`; archive member `./data40.json` record 51428.
- Record ID: `archive:./data40.json:51428:2807b1310e21286191251e0b344fca98f89ccd17620a9184fc367d619be184ba`.
- Exact-story SHA-256: `bb5637cdbc583c0006af5b05a82fcbc0f48848f771802f990bc94d46e356319b`.

>
> Once there was a hero. He was very brave and loved to sing all day. One day he had an adventure. He went to a field full of trees and flowers. The sun was shining and he felt so dizzy from all of the beauty! Suddenly he heard a noise! It was a singing bird. He looked around and found a pin on the ground. He picked it up and sang a happy song with the bird. He was so happy and couldn't believe his luck.
> The bird and the hero were friends now. They sang and played in the field every day. The hero kept his pin as a reminder of the day they met. The hero would never forget the time he spent with his new bird friend, singing and being dizzy in the sunshine.

#### `control/B/validation` — same verb column

- Recipe: adjective `dependable`, noun `view`, verb `observe`; features `conflict, dialogue`.
- Recipe cell: `N1 × V2`; control arm `same verb column`.
- Length: 159 tokens; condition lower median 159 tokens.
- Source: `GPT-4`; archive member `./data28.json` record 49531.
- Record ID: `archive:./data28.json:49531:7b5fc357beae53cedaca1f528f2efebdccb1a9ce762bb7cc51fad9f76c18551d`.
- Exact-story SHA-256: `8df8683e0645ac3747300e13c910f21e7eac9d3e6d3d33db997c22e9cb3c7584`.

> One day, a dependable dog named Spot and his friend, a little boy named Tim, went to the park. They liked to observe the view of the trees, flowers, and birds. They played together every day.
> While they were playing, a big cat came and said, "This is my park. You must leave!" Tim and Spot did not like this. They wanted to stay and play. Tim said to the cat, "No, we want to play here too. We can all play together."
> The cat thought for a moment and said, "Okay, we can all play together." So, Tim, Spot, and the big cat played in the park. They had lots of fun and became good friends. The park was a happy place for everyone.

### World C: N7 × V3

This world contains recipes pairing any of 133 nouns in `N7` with any of 49 verbs in `V3`. Its matched control combines an equal-count same-noun-row arm and same-verb-column arm from held-in validation.

#### `world/C/validation`

- Recipe: adjective `great`, noun `scarf`, verb `stir`; features `conflict, moralvalue`.
- Recipe cell: `N7 × V3`.
- Length: 178 tokens; condition lower median 178 tokens.
- Source: `GPT-4`; archive member `./data02.json` record 5328.
- Record ID: `archive:./data02.json:5328:29e826bdd75e2b010a7da5ee9088dc2a20fc157dfc5cd8960184275b3ba063c1`.
- Exact-story SHA-256: `eae9fee9465821f2f26f6aef67defccea985371f4720a7b840bad049db4d342e`.

> One day, a little girl named Sue found a great big scarf. She loved the scarf and wanted to wear it. Sue put the scarf around her neck and went outside to play. She saw her friend Tim playing with a big stick. Tim was using the stick to stir a big pile of leaves.
> Sue wanted to help Tim stir the leaves, but her scarf got caught in the pile. The scarf pulled her down and she fell into the leaves. Sue was sad because her great big scarf was dirty. Tim stopped stirring the leaves and helped Sue get up.
> Tim had an idea. He took Sue's scarf and washed it in a bucket of water. The scarf was clean again. Sue was happy and thanked Tim. They both learned that helping each other is important. They played together and stirred the leaves with their big sticks, having fun and being good friends.

#### `control/C/validation` — same noun row

- Recipe: adjective `perfect`, noun `pedal`, verb `have`; features `foreshadowing`.
- Recipe cell: `N7 × V0`; control arm `same noun row`.
- Length: 183 tokens; condition lower median 183 tokens.
- Source: `GPT-4`; archive member `./data46.json` record 55631.
- Record ID: `archive:./data46.json:55631:78f2b5e9a222da70e75cd749771b31007126ab4230105288af730e20fa1b5112`.
- Exact-story SHA-256: `2f026f860c149aa2c93387e8dfce0f2413943764bf395e5e04590fee032d2ce1`.

>
> Once upon a time there was a girl. She had a tricycle - it was bright red. She loved it more than anything. Every day, she would go out and ride it. She would pedal fast, and it made her happy.
> One day, as she rode, she saw something new. It was a perfect bike - shiny, green, and fancy. It was just what she wanted. She looked at it and smiled.
> The next day, she went back. It was just as she remembered. It was perfect - she knew she had to have it. She saw the salesperson, and asked to buy it.
> The salesperson smiled and said yes. She was so excited. She rode it home. She couldn't believe that it was hers! She was so happy. She was going to have the perfect bike. She was able to pedal faster than ever before!

#### `control/C/validation` — same verb column

- Recipe: adjective `huge`, noun `task`, verb `zoom`; features `moralvalue`.
- Recipe cell: `N0 × V3`; control arm `same verb column`.
- Length: 183 tokens; condition lower median 183 tokens.
- Source: `GPT-4`; archive member `./data16.json` record 65685.
- Record ID: `archive:./data16.json:65685:6603dece3698bd2c809ccafc28fff7b0be1093cd77746ac53e0da579fed12513`.
- Exact-story SHA-256: `2c46bacb7aa0a6789359ffea9def7e95cf6a784572df3eff2a8784088bb2aceb`.

>
> Once upon a time, there lived a family of ants. Every day, they had a lot of tasks to do and they worked very hard. They would zoom up and down, trying to finish everything they could.
> One day, they saw something very huge! It was a big piece of honey, and they all wanted to enjoy it. But they knew it would take a very long time to carry it away.
> So, one of the ants had an idea. He said, "Let's all work together and zoom as fast as we can! Then we can all joyful enjoy this sweet honey".
> So, they all worked together and zoomed around. And, in no time, the honey was brought back to the anthill!
> The moral of the story is that working together for a common task is the best way to succeed. Working together is much more powerful than working alone!

### World D: N6 × V3

This world contains recipes pairing any of 133 nouns in `N6` with any of 49 verbs in `V3`. Its matched control combines an equal-count same-noun-row arm and same-verb-column arm from held-in validation.

#### `world/D/validation`

- Recipe: adjective `modern`, noun `pastel`, verb `stir`; features `dialogue`.
- Recipe cell: `N6 × V3`.
- Length: 178 tokens; condition lower median 178 tokens.
- Source: `GPT-3.5`; archive member `./data39.json` record 1813.
- Record ID: `archive:./data39.json:1813:636f009a1402f6849370cc36bdbc8620862de30bd836a4fb19606371a9523e4a`.
- Exact-story SHA-256: `76c62b797e0f958c083167b63c2b7e7c76f275245bc97956589d85d12b929f87`.

> Once upon a time, there was a little boy named Timmy. Timmy loved to draw and he loved pastels. He had a big box of pastels with many colors. One day, Timmy's mom bought him a modern pastel set. Timmy was very happy.
> Timmy's mom asked, "Would you like to draw with your new pastels, Timmy?"
> "Yes, please!" said Timmy.
> Timmy picked a pastel and started to draw. He drew a picture of a cat. His mom asked, "Can I help you stir the colors, Timmy?"
> Timmy said, "Yes, please!" and his mom helped him stir the pastels. Timmy's picture of the cat was very pretty with the modern pastels. Timmy and his mom were very happy with the picture.

#### `control/D/validation` — same noun row

- Recipe: adjective `safe`, noun `rainbow`, verb `scream`; features `dialogue`.
- Recipe cell: `N6 × V6`; control arm `same noun row`.
- Length: 160 tokens; condition lower median 160 tokens.
- Source: `GPT-3.5`; archive member `./data38.json` record 46020.
- Record ID: `archive:./data38.json:46020:06469bd549bd548df87affa597e830d9dd1954f23510d7c869b8964d42d903e4`.
- Exact-story SHA-256: `a7154fb45491471aa86f8e27457893c9bafb75aa230a0eae03acdb73cc17c5fc`.

> Once upon a time, there was a little girl named Lily. She loved to play outside in the grass and watch the birds fly. One day, Lily saw a big, beautiful rainbow in the sky. She pointed at it and said, "Look, Mama! Rainbow!"
> Her mother smiled and said, "Yes, Lily, that's a rainbow. It's very pretty." Suddenly, the sky turned dark and it started to rain. Lily got scared and ran to her mother, screaming. "Mama, I'm scared!" she cried.
> "Don't worry, Lily," her mother said. "We can go inside where it's safe and dry." They went inside and watched the rain fall outside. Lily felt much better knowing she was safe and warm with her mother.

#### `control/D/validation` — same verb column

- Recipe: adjective `tough`, noun `log`, verb `receive`; features `none`.
- Recipe cell: `N4 × V3`; control arm `same verb column`.
- Length: 160 tokens; condition lower median 160 tokens.
- Source: `GPT-3.5`; archive member `./data00.json` record 60394.
- Record ID: `archive:./data00.json:60394:c7c049084228eb1291b915d03110cd46559f3ddeedec051f24c85eb630785b71`.
- Exact-story SHA-256: `d37a51ff909c884c17adb53d7181c232b347be990190e8d6fd63efad12781d2b`.

> Once upon a time, there was a little boy named Timmy. Timmy loved to play outside in the woods. One day, Timmy found a big log on the ground. It was tough to move, but Timmy was strong. He pushed and pulled until he got the log to his house.
> When Timmy got home, his mom was happy to see him. She gave him a big hug and a kiss. Timmy was so happy to receive love from his mom. He showed her the log he found and she was impressed.
> Timmy and his mom decided to use the log as a seat in their backyard. They sat on the tough log and enjoyed the beautiful day. Timmy was proud of himself for finding the log and making his mom happy.

### World E: N2 × V4

This world contains recipes pairing any of 134 nouns in `N2` with any of 49 verbs in `V4`. Its matched control combines an equal-count same-noun-row arm and same-verb-column arm from held-in validation.

#### `world/E/validation`

- Recipe: adjective `original`, noun `tap`, verb `promise`; features `none`.
- Recipe cell: `N2 × V4`.
- Length: 177 tokens; condition lower median 177 tokens.
- Source: `GPT-3.5`; archive member `./data44.json` record 45919.
- Record ID: `archive:./data44.json:45919:6be605ede2b41576b71b9a86ca4fac0116739bdec9746892cc0d2a3b41cca691`.
- Exact-story SHA-256: `338956d0f035115b4c5ff2fbe90e7a20fc0db2181f5230e5993d89d6e6b946e2`.

> Once upon a time, there was a little girl named Lily. She loved to play with her toys, especially her teddy bear. One day, Lily's mom took her to a toy store. Lily saw a shiny new toy car that she really wanted, but her mom said they couldn't buy it because it was too expensive.
> Lily was sad, but her mom promised her that they would come back to the store another day and buy something special. As they were leaving the store, Lily saw an original tap dancer performing outside. She was so amazed by the dancer's fancy footwork and shiny shoes.
> When they got home, Lily's mom surprised her with a tap dancing toy bear. It wasn't the toy car Lily had wanted, but it was just as special and original. From that day on, Lily and her tap dancing bear were inseparable.

#### `control/E/validation` — same noun row

- Recipe: adjective `thin`, noun `infant`, verb `bow`; features `conflict, twist`.
- Recipe cell: `N2 × V2`; control arm `same noun row`.
- Length: 182 tokens; condition lower median 182 tokens.
- Source: `GPT-4`; archive member `./data32.json` record 49883.
- Record ID: `archive:./data32.json:49883:144414f68aa2500850388368a48dd61eec1874c5acecd2dc1a1ec93062bd0809`.
- Exact-story SHA-256: `87beab148953b5325718c6bbd20232a45e647e4f8e8b5f6341a4d18a2ccdaef2`.

> Once upon a time, there was a thin cat. The cat liked to bow to everyone it met. One day, the cat saw an infant. The infant was crying very loud. The cat wanted to help, so it bowed to the infant, but the crying did not stop.
> The cat thought very hard. It had an idea. The cat ran away to find a toy. It brought the toy back to the infant. The cat bowed and gave the toy to the infant. But the infant still cried.
> Then, something unexpected happened. A big dog came. The dog looked mean. The cat was scared, but it did not want the infant to be sad. The cat bowed to the dog. The dog was surprised. It stopped being mean and started to play with the infant. The infant stopped crying and laughed. The cat, the dog, and the infant became good friends.

#### `control/E/validation` — same verb column

- Recipe: adjective `unknown`, noun `glue`, verb `accept`; features `none`.
- Recipe cell: `N7 × V4`; control arm `same verb column`.
- Length: 182 tokens; condition lower median 182 tokens.
- Source: `GPT-3.5`; archive member `./data45.json` record 95821.
- Record ID: `archive:./data45.json:95821:8031736698898f360793fa5d3e624e34357fcdc1b0e6d14f0750652c4ce78085`.
- Exact-story SHA-256: `c0cb4d912ec07006c51b595f4d8d4626faf9058da842e006392e84f92f85b60f`.

> Once upon a time, there was a boy named Timmy. Timmy loved to play with his toys all day long. One day, Timmy's mom asked him to help her make a craft. She gave him some glue and paper to stick together. Timmy was happy to help and accept the task. He carefully put the glue on the paper and stuck them together.
> Suddenly, Timmy noticed that some of the glue was on his hand. He tried to wipe it off, but it wouldn't come off. He felt a little scared because he didn't know what was happening. His mom told him not to worry because it was just some unknown glue that would come off in the bath later.
> Timmy felt relieved and continued to help his mom with the craft. He learned that sometimes things can be unknown, but it's okay to ask for help and accept it.

## Fresh 6×6 fallback calibration

Partition: `7bf90c70ca7207d8b0fdd7896eed7a2ae019bbcbd74126cfcc2115ae0759b4fb`.

### Condition inventory

| Validation condition | Occurrences | Active tokens | Lower-median length | Samples below |
| --- | ---: | ---: | ---: | ---: |
| `base/validation` | 128,116 | 24,393,576 | 178 | 1 |
| `world/A/validation` | 13,905 | 2,644,924 | 177 | 1 |
| `world/B/validation` | 13,899 | 2,645,989 | 178 | 1 |
| `world/C/validation` | 13,942 | 2,651,514 | 178 | 1 |
| `world/D/validation` | 13,907 | 2,644,328 | 178 | 1 |
| `world/E/validation` | 13,892 | 2,644,022 | 178 | 1 |
| `control/A/validation` | 13,905 | 2,644,924 | 182 | 2 |
| `control/B/validation` | 13,899 | 2,639,394 | 159 | 2 |
| `control/C/validation` | 13,942 | 2,651,514 | 182 | 2 |
| `control/D/validation` | 13,907 | 2,637,735 | 159 | 2 |
| `control/E/validation` | 13,892 | 2,644,022 | 182 | 2 |

### Selected world topology

| World | Physical cell | Eligible groups | Active tokens |
| :---: | :---: | ---: | ---: |
| A | `N1 × V2` | 139,066 | 26,457,390 |
| B | `N2 × V2` | 139,007 | 26,468,038 |
| C | `N2 × V3` | 139,423 | 26,522,154 |
| D | `N1 × V3` | 139,079 | 26,451,803 |
| E | `N3 × V0` | 138,937 | 26,448,667 |

### What the selected buckets contain

Examples are 12 evenly spaced entries from each alphabetically sorted bucket, not hand-picked themes.

| Bucket | Kind | Words | Archive token mass | Deterministic examples |
| :---: | :---: | ---: | ---: | --- |
| `N1` | noun | 178 | 157,855,009 | `airport`, `cat`, `dust`, `goat`, `leg`, `musician`, `pastry`, `raven`, `sight`, `surf`, `victory`, `young` |
| `N2` | noun | 178 | 157,856,774 | `angel`, `candle`, `dog`, `gem`, `inch`, `mask`, `oyster`, `planet`, `shoe`, `sunset`, `underground`, `yoga` |
| `N3` | noun | 178 | 157,855,620 | `aeroplane`, `broccoli`, `comet`, `fairy`, `jungle`, `map`, `network`, `ranch`, `slide`, `swing`, `uniform`, `zipper` |
| `V0` | verb | 66 | 158,335,048 | `arrive`, `collect`, `dive`, `fold`, `invite`, `listen`, `offer`, `punish`, `set`, `speak`, `twist`, `yawn` |
| `V2` | verb | 66 | 158,340,502 | `admire`, `celebrate`, `eat`, `increase`, `lose`, `poke`, `quarrel`, `remain`, `seem`, `strike`, `tie`, `worry` |
| `V3` | verb | 66 | 158,338,916 | `add`, `care`, `excite`, `ignore`, `pack`, `race`, `request`, `serve`, `spin`, `suffer`, `vanish`, `write` |

### Held-in base validation

#### `base/validation`

- Recipe: adjective `peaceful`, noun `attic`, verb `choose`; features `dialogue, moralvalue`.
- Recipe cell: `N2 × V1`.
- Length: 178 tokens; condition lower median 178 tokens.
- Source: `GPT-4`; archive member `./data29.json` record 83068.
- Record ID: `archive:./data29.json:83068:23a5d5b47962d965b9a03c491dae1746c56d7f3189ea98021babc49193539ac2`.
- Exact-story SHA-256: `97f0f2cd7c97cd2d0ba2c17242b3c1fd40f281f5df429b04b0c9e4223196c3a4`.

> Once upon a time, there was a little girl named Emma. She loved to play with her toys in her room. One day, her mom told her to choose a toy to give away. Emma was sad because she loved all her toys.
> Emma's mom said, "Let's go to the attic and find a toy you don't play with anymore." They went to the peaceful attic and looked at all the toys. Emma saw a teddy bear she had not played with for a long time.
> Emma said, "I will give this teddy bear away." She took the teddy bear to her friend, Lily. Lily was very happy and said, "Thank you, Emma!" Emma felt good about giving her teddy bear to someone who would love it.
> The moral of the story is that sharing and giving can make you and others happy.

### World A: N1 × V2

This world contains recipes pairing any of 178 nouns in `N1` with any of 66 verbs in `V2`. Its matched control combines an equal-count same-noun-row arm and same-verb-column arm from held-in validation.

#### `world/A/validation`

- Recipe: adjective `rude`, noun `teach`, verb `pour`; features `twist`.
- Recipe cell: `N1 × V2`.
- Length: 177 tokens; condition lower median 177 tokens.
- Source: `GPT-4`; archive member `./data13.json` record 62544.
- Record ID: `archive:./data13.json:62544:175d2b442de65af7d7b00550187b8707ad406efa5d6195cfe9534d6a69b612d0`.
- Exact-story SHA-256: `244061de347c6ea177400763db5e38281a8a3a727017ce0f67ad796d75b9ac86`.

> One day, a girl named Amy went to school. At school, she had a teach who was very nice. Her teach showed her how to pour water into a cup. Amy was happy to learn this. She liked to pour water and not be rude, like some kids at school.
> One day, while Amy was pouring water, a cat came into the room. The cat was very rude! It jumped on the table and knocked over the cup. Water went everywhere! Amy was sad, but the teach told her it was okay. They cleaned up the water together.
> Then, something unexpected happened. The rude cat started to pour water too! It used its paw to pour water into a cup. Amy and the teach were very surprised. The cat was not rude anymore. It wanted to learn, just like Amy. They all became friends and poured water together.

#### `control/A/validation` — same noun row

- Recipe: adjective `bright`, noun `towel`, verb `accept`; features `dialogue, twist`.
- Recipe cell: `N1 × V1`; control arm `same noun row`.
- Length: 182 tokens; condition lower median 182 tokens.
- Source: `GPT-4`; archive member `./data26.json` record 19980.
- Record ID: `archive:./data26.json:19980:352bef2d358c93fa9fc6d06622bbf3a4ab0822dfda8b0d8ded7a443fdab096d7`.
- Exact-story SHA-256: `b167210542850d720ba58ee3a1937537ef20088a280700bb7a90fb50de9b7bed`.

> One day, a bright sun shone in the sky. A boy named Tim went to the park. He took a towel with him. He wanted to sit on the towel and eat his lunch.
> At the park, Tim met a girl named Sue. She saw his towel and said, "That is a nice towel! Can I sit with you?" Tim said, "Yes, you can!" They sat on the towel and ate their lunch together.
> As they ate, a big wind came. It blew Tim's towel away! Tim and Sue ran after it. They saw a dog with the towel in its mouth. The dog wagged its tail and came to them. Tim said, "Please give me my towel back." The dog dropped the towel and Tim accepted it. Tim and Sue laughed and played with the dog. They had a fun day at the park with their new friend.

#### `control/A/validation` — same verb column

- Recipe: adjective `cheap`, noun `pit`, verb `run`; features `none`.
- Recipe cell: `N4 × V2`; control arm `same verb column`.
- Length: 182 tokens; condition lower median 182 tokens.
- Source: `GPT-4`; archive member `./data14.json` record 28406.
- Record ID: `archive:./data14.json:28406:b70e6d5fb11674e53f7ba489ef6a242d31d6ddf5df090395e0eda5eb22c8807e`.
- Exact-story SHA-256: `e5090ee14c0039a59cd5b46075efc4af1ecb0e8696d6b6643ed53a20abc05c7d`.

>
> Once there was a boy who loved to run. Every day he put on his shoes and ran around the park. One day he saw a big pit and ran towards it. He was amazed. It was so deep and he wanted to explore it. He quickly scrambled down into the deep pit.
> At the bottom of the pit was a big, cheap treasure. The boy was so excited and started to collect it. He gathered all the cheap treasure he could find and ran back up out of the pit. He was filled with joy, happy to have found such a great treasure in the pit.
> The boy ran back home with his treasure. He told all of his friends what he had found in the pit and they all ran around the park together looking for other deep pits with treasure. They never found another one but the boy never forgot that wonderful day he found his treasure in the pit.

### World B: N2 × V2

This world contains recipes pairing any of 178 nouns in `N2` with any of 66 verbs in `V2`. Its matched control combines an equal-count same-noun-row arm and same-verb-column arm from held-in validation.

#### `world/B/validation`

- Recipe: adjective `pale`, noun `song`, verb `raise`; features `twist`.
- Recipe cell: `N2 × V2`.
- Length: 178 tokens; condition lower median 178 tokens.
- Source: `GPT-4`; archive member `./data43.json` record 47419.
- Record ID: `archive:./data43.json:47419:c431dbafce5d9e04bbf33daeff888752eee38f174b4d28138969a449fda8a3ab`.
- Exact-story SHA-256: `83bb1be895ce6e27822bbfe741a36e0735da340c4b8e4eb76368790e2dfb52f1`.

> Once upon a time, there was a pale bird named Blue. Blue loved to sing songs. Every day, Blue would raise her voice and sing for all her friends in the forest.
> One day, Blue was singing her song when she saw a big box. Blue was curious and wanted to see what was inside. She flew down and opened the box with her beak. To her surprise, she found a shiny, magic hat inside.
> Blue put the hat on her head and started to sing her song again. But this time, something unexpected happened. As Blue sang her song, all her friends in the forest started to dance! They danced and danced as long as Blue sang her song. Blue and her friends had the best time dancing and singing in the forest. From that day on, Blue's song became even more special, and everyone loved to dance to it.

#### `control/B/validation` — same noun row

- Recipe: adjective `tidy`, noun `log`, verb `yawn`; features `badending, dialogue`.
- Recipe cell: `N2 × V0`; control arm `same noun row`.
- Length: 159 tokens; condition lower median 159 tokens.
- Source: `GPT-4`; archive member `./data30.json` record 12898.
- Record ID: `archive:./data30.json:12898:9f86c25513218b4aa4372e2cbf6e09a99d5ddbd57aed8093ed015a7f675dc5ec`.
- Exact-story SHA-256: `5abf1a38340e2c2662f37f3f1307eb4316d12cc375b8c5c9d2c4f622a7d02586`.

> One day, a big bear named Bob yawned as he woke up in his tidy cave. He felt hungry, so he went outside to look for food. He saw a log and thought it might have bugs inside to eat.
> Bob said to his friend, the rabbit, "I am going to eat bugs from this log. Do you want some too?" The rabbit said, "No, thank you. I like to eat grass." So, Bob tried to break the log to find the bugs.
> But the log was very hard, and Bob could not break it. He tried and tried, but he got very tired. At last, the log fell on Bob's foot, and it hurt him a lot. He cried and went back to his cave without any food.

#### `control/B/validation` — same verb column

- Recipe: adjective `bright`, noun `chalk`, verb `examine`; features `dialogue, twist`.
- Recipe cell: `N0 × V2`; control arm `same verb column`.
- Length: 159 tokens; condition lower median 159 tokens.
- Source: `GPT-4`; archive member `./data38.json` record 37637.
- Record ID: `archive:./data38.json:37637:ff11f761a7a75a997e1afff990f1c1bfa267e0f650b549861f7988bc5a04faff`.
- Exact-story SHA-256: `29ab29cb7d83aa14159a10120973cd3fb410c69f2a64a40d00b2c99ad3e07c42`.

> One day, a boy named Tom found a bright piece of chalk. He wanted to draw on the sidewalk. He showed his friend, Sue. "Look! I have chalk!" Tom said.
> Sue wanted to draw too, so they both started to draw. They drew a big sun, a house, and a tree. They were having fun. But then, something unexpected happened. The drawings started to move!
> Tom and Sue were scared. They didn't know what to do. They decided to examine the moving drawings. They saw the sun was smiling, the house was dancing, and the tree was waving. The drawings were friendly!
> Tom and Sue laughed and played with their new drawing friends. They had a fun day with the bright chalk and their moving drawings.

### World C: N2 × V3

This world contains recipes pairing any of 178 nouns in `N2` with any of 66 verbs in `V3`. Its matched control combines an equal-count same-noun-row arm and same-verb-column arm from held-in validation.

#### `world/C/validation`

- Recipe: adjective `stubborn`, noun `order`, verb `suffer`; features `twist`.
- Recipe cell: `N2 × V3`.
- Length: 178 tokens; condition lower median 178 tokens.
- Source: `GPT-4`; archive member `./data30.json` record 92941.
- Record ID: `archive:./data30.json:92941:9c1e0bc2f898b71b85020f51e98797bdebf197c6be52a6ec7016627b2345eba5`.
- Exact-story SHA-256: `dcc5f48f6be04d1ee465dcdc36af0e5e8c32b12b95a085885e9c2a44c781af69`.

> Once upon a time, there was a stubborn cat named Kitty. Kitty did not like to follow the order that her mom, Mama Cat, gave her. Mama Cat wanted Kitty to clean her room, but Kitty did not want to.
> One day, Kitty went to play outside. She saw a big tree and wanted to climb it. But she had never climbed a tree before. Kitty was scared she would suffer if she fell. But she was stubborn, so she started climbing.
> Kitty climbed higher and higher. Suddenly, she saw a big bird on a branch. The bird said, "I can help you get down if you promise to follow orders from your mom." Kitty thought about it and agreed. The bird helped Kitty get down safely. From that day on, Kitty was not stubborn anymore and followed the orders from her mom. And they lived happily ever after.

#### `control/C/validation` — same noun row

- Recipe: adjective `careless`, noun `bee`, verb `reveal`; features `dialogue`.
- Recipe cell: `N2 × V5`; control arm `same noun row`.
- Length: 182 tokens; condition lower median 182 tokens.
- Source: `GPT-3.5`; archive member `./data03.json` record 91346.
- Record ID: `archive:./data03.json:91346:1dfaef70be1b855151cc76044b7730e85231c011277090bdd2521036910795b5`.
- Exact-story SHA-256: `2a6d601806f4959e612bc3f4899925fc7f09649bba16c3398c0cf161298128d2`.

> Once upon a time, there was a little girl named Lily. She loved playing outside, especially in her garden. One day, she saw a bee buzzing around the flowers. "Hi, little bee! What are you doing?" she asked.
> "I'm looking for some nectar to bring back to my hive," replied the bee.
> Lily was so excited to see the bee up close. She reached out to touch it, but she was careless and accidentally squished the bee. She felt so bad and didn't know what to do.
> Suddenly, her mom came outside and saw what had happened. She gently picked up the bee and revealed that it was still alive. "We need to be very careful around bees, they are very important for our flowers and plants," she said. From that day on, Lily learned to be more careful and always treated the bees with kindness.

#### `control/C/validation` — same verb column

- Recipe: adjective `bald`, noun `pencil`, verb `spin`; features `dialogue`.
- Recipe cell: `N5 × V3`; control arm `same verb column`.
- Length: 182 tokens; condition lower median 182 tokens.
- Source: `GPT-3.5`; archive member `./data39.json` record 8377.
- Record ID: `archive:./data39.json:8377:47b3dcad911ab0490ddb82a0bd8701ab046e43b80d09ed9fc12b9ad0aa045ce2`.
- Exact-story SHA-256: `7aa9c2590eeb5ea44c0dca69a70f6d3d0d916c93aa8ca8001894066b91a479f1`.

> Once upon a time, there was a boy named Timmy. Timmy loved to draw with his pencil. One day, he saw his dad's bald head and thought it looked funny. "Daddy, why is your head bald?" asked Timmy. "I don't know, son. It just is," replied his dad.
> Later that day, Timmy was playing with his toy top. He watched it spin round and round. "Wow, it's spinning so fast!" he shouted. His dad came over and said, "That's called spinning. Do you want to try it with your pencil?" Timmy nodded and tried spinning his pencil on the table. It was so much fun!
> From then on, Timmy loved to spin his pencil and watch it go round and round. And whenever he saw his dad's bald head, he couldn't help but giggle.

### World D: N1 × V3

This world contains recipes pairing any of 178 nouns in `N1` with any of 66 verbs in `V3`. Its matched control combines an equal-count same-noun-row arm and same-verb-column arm from held-in validation.

#### `world/D/validation`

- Recipe: adjective `scared`, noun `circle`, verb `open`; features `conflict, dialogue`.
- Recipe cell: `N1 × V3`.
- Length: 178 tokens; condition lower median 178 tokens.
- Source: `GPT-4`; archive member `./data33.json` record 45999.
- Record ID: `archive:./data33.json:45999:1c74128e2155cb016d1e8f1c2e7f18d2a125f6bce9423f474b8b0c74528aa1f6`.
- Exact-story SHA-256: `f51874b22270030c4f52924edfc55cfffa5b4e69c5cf56d809cdac475c2fa339`.

>
> Jack was scared. He had never seen anything like this before.
> He was looking at a big circle on the ground. He had never seen a circle so big before.
> Jack didn't know what to do. He was so scared.
> Suddenly, the circle opened and out jumped a big, scary monster. Jack screamed and ran away.
> He ran and ran until he came to his house. He opened the door and ran inside, screaming.
> His mom asked, "What is it Jack? What happened?"
> "A monster!" he said, "A monster jumped out of a big circle! I'm scared!"
> His mom hugged him and said, "Don't be scared Jack. That was just a nightmare. Everything is okay."
> Jack smiled. He knew his mom was right. He was safe at home and the scary monster was gone.

#### `control/D/validation` — same noun row

- Recipe: adjective `charming`, noun `cooler`, verb `please`; features `dialogue, twist`.
- Recipe cell: `N1 × V5`; control arm `same noun row`.
- Length: 159 tokens; condition lower median 159 tokens.
- Source: `GPT-4`; archive member `./data24.json` record 78766.
- Record ID: `archive:./data24.json:78766:6f7229552f86b8343efe388166ced10952755b0ba3dde8b00f9b679b1338cd13`.
- Exact-story SHA-256: `bc85e26d77ab302423df12f85ae6da32308b45c6d79801339c913ac6d1f397a9`.

> One day, a charming little dog named Max went for a walk. He saw a cooler under a big tree. Max was very hot, so he thought, "Maybe there is something cold to drink in the cooler!"
> Max went to the cooler and tried to open it, but it was too hard for him. Just then, a big friendly bear named Ben walked by. Max said, "Please, Ben, can you help me open the cooler?" Ben smiled and said, "Of course, little friend!"
> Ben used his big paws to open the cooler. But instead of cold drinks, they found a big pile of yummy ice cream! Max and Ben were so surprised and happy. They sat under the tree and shared the ice cream, becoming the best of friends.

#### `control/D/validation` — same verb column

- Recipe: adjective `elderly`, noun `onion`, verb `scare`; features `conflict, dialogue`.
- Recipe cell: `N3 × V3`; control arm `same verb column`.
- Length: 159 tokens; condition lower median 159 tokens.
- Source: `GPT-4`; archive member `./data29.json` record 22205.
- Record ID: `archive:./data29.json:22205:aec34bf42dd6c383d93f51d072228eca26d27d7c65e76652151627c9ec183458`.
- Exact-story SHA-256: `d723c1cbfc3aa0eb336ebc7ccbf03888df3a0b9b5f69ccf4feaa6ea1335c4dff`.

> Once upon a time, there was an elderly lady named Mary. She had a big garden with many plants. In her garden, she grew a very big onion. One day, a sneaky rabbit came into the garden. The rabbit wanted to eat the big onion.
> Mary saw the rabbit and said, "No, rabbit! You cannot eat my big onion!" The rabbit did not listen. So, Mary thought of a plan to scare the rabbit away. She put a big, scary mask near the onion.
> When the rabbit saw the scary mask, it said, "Oh no! I am scared!" The rabbit ran away very fast. It did not eat the big onion. Mary was happy that her plan worked. She could now keep her big onion safe in her garden.

### World E: N3 × V0

This world contains recipes pairing any of 178 nouns in `N3` with any of 66 verbs in `V0`. Its matched control combines an equal-count same-noun-row arm and same-verb-column arm from held-in validation.

#### `world/E/validation`

- Recipe: adjective `thin`, noun `diary`, verb `organize`; features `conflict`.
- Recipe cell: `N3 × V0`.
- Length: 178 tokens; condition lower median 178 tokens.
- Source: `GPT-4`; archive member `./data13.json` record 25614.
- Record ID: `archive:./data13.json:25614:0d4bd8e8ec89e63bb163690301446699f09b227568628a1ffeee9f7180b80c99`.
- Exact-story SHA-256: `3f448429c6873dee99050250b8a1800822676054ddc8161cff7f512f9f5e8ce7`.

>
> Once upon a time there was a little girl who wanted to organize her things. She had a plan. She wanted to use her diary to keep track of important dates. Every day she tried to write in it, but it was so thin! Day by day, she became increasingly frustrated and angry. Finally she decided to look for a thicker diary. The next morning, she went out and found a diary that was much thicker.
> But it was too thick too fit inside her school bag! She thought hard and got an idea. She grabbed some colored paper, scissors and glue and carefully cut, folded and stuck them together to make a little pocket. Then, she slide the diary into it and put everything in her backpack. She smiled to herself, feeling relieved and proud of her work. From then on, she never forget to organize her diary and everything else!

#### `control/E/validation` — same noun row

- Recipe: adjective `soft`, noun `village`, verb `shine`; features `moralvalue`.
- Recipe cell: `N3 × V3`; control arm `same noun row`.
- Length: 182 tokens; condition lower median 182 tokens.
- Source: `GPT-3.5`; archive member `./data13.json` record 33287.
- Record ID: `archive:./data13.json:33287:955f22284645d321e57047aa6db9b9ba7f1da459514d892a81faa1fe059de2a6`.
- Exact-story SHA-256: `6b1f2206990011a7f0851b3cfd3f44a67c3a41e5c2ff1d5bfe98d82a83a7ae2d`.

> Once upon a time, there was a village on a hill. The people in the village were very happy. They played and laughed all day long. One day, a little girl named Lily came to the village. She had a soft heart and wanted to help the people in the village.
> Lily saw that the village was very dark at night. The stars were shining bright, but the village was still dark. So, Lily decided to bring light to the village. She went to the top of the hill and put a big light there. The light shone bright and lit up the whole village.
> The people in the village were so happy. They thanked Lily for bringing light to their village. Lily felt very happy too. She learned that even a small person can make a big difference. She also learned that when you shine your light, you can help others and make them happy.

#### `control/E/validation` — same verb column

- Recipe: adjective `yummy`, noun `toy`, verb `laugh`; features `none`.
- Recipe cell: `N2 × V0`; control arm `same verb column`.
- Length: 182 tokens; condition lower median 182 tokens.
- Source: `GPT-3.5`; archive member `./data24.json` record 88411.
- Record ID: `archive:./data24.json:88411:fb6cc125a69a17830319950a93cf1e000de5e1ca48eda6a1b7e5dc5bf0345357`.
- Exact-story SHA-256: `ea2bac083bd7a3bb8f4f500071a09972e7f5fda7f1a8b787b2a3c0ab2ff769ce`.

> Once upon a time, there was a little boy named Timmy. Timmy loved to play with his toys. His favorite toy was a big red truck that he would push around and make vroom vroom noises.
> One day, Timmy's mommy made him a yummy sandwich for lunch. It had peanut butter and jelly on it, just the way Timmy liked it. He took a big bite and smiled. "Mmm, yummy!" he said.
> Suddenly, Timmy's daddy came in and tickled him. Timmy laughed and dropped his sandwich. His dog ran over and started to lick up the peanut butter and jelly. Timmy laughed even harder and grabbed his toy truck. He pushed it around the room, chasing after his dog and playing with his daddy. It was a happy day full of laughter and fun with his favorite toy. The end.
