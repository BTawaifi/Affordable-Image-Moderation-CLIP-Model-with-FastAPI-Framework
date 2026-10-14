import { Column, Entity, PrimaryGeneratedColumn, ManyToOne } from 'typeorm';
import { ClassificationResult } from './classification-results.entity';

@Entity()
export class LabelProbability {
  @PrimaryGeneratedColumn()
  id: number;

  @Column()
  classificationResultId: number;

  @ManyToOne(() => ClassificationResult, (result) => result.id, {
    onDelete: 'CASCADE',
  })
  classificationResult: ClassificationResult;

  @Column()
  label: string;

  @Column('float')
  probability: number;
}
