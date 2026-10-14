// moderation-result.entity.ts
import { Column, Entity, OneToMany, PrimaryGeneratedColumn } from 'typeorm';
import { LabelProbability } from './label-probabilities.entity';

@Entity()
export class ClassificationResult {
  @PrimaryGeneratedColumn()
  id: number;

  @Column()
  taskId: string;

  @Column()
  userId: number;

  @Column()
  classifier: string;

  @Column()
  location: string;

  @Column()
  prediction: string;

  @Column({ default: false })
  moderated: boolean;

  @Column({ type: 'timestamp', default: () => 'CURRENT_TIMESTAMP' })
  createdAt: Date;

  @Column({ type: 'timestamp', onUpdate: 'CURRENT_TIMESTAMP', nullable: true })
  modifiedAt: Date | null;

  @OneToMany(
    () => LabelProbability,
    (labelProbability) => labelProbability.classificationResult,
  )
  labelProbability: LabelProbability[]; // Updated property name
}
