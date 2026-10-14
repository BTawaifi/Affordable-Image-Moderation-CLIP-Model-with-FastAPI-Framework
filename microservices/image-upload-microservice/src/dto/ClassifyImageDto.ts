import { IsOptional, IsString } from 'class-validator';
import { Transform, Type } from 'class-transformer';

export class ClassifyImageDto {
  @Transform(({ value }) => value.split(',').map((item: string) => item.trim()))
  @Type(() => String)
  @IsOptional()
  @IsString({ each: true })
  classes: string;
}
